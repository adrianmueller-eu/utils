import sys, psutil
from contextlib import contextmanager
from math import prod
import numpy as np
try:
    import scipy.sparse as sp
except ImportError:
    pass

from .constants import *
from .state import partial_trace, ket, dm, unket, count_qubits, random_ket, random_dm, plotQ, is_state, ensemble_from_state
from .hamiltonian import parse_hamiltonian
from .info import *
from .unitary import parse_unitary, get_unitary, Fourier_matrix
from ..mathlib import choice, normalize, binstr_from_int, bipartitions, is_unitary, is_hermitian, is_diag, trace_product, eigh
from ..plot import imshow
from ..utils import is_int, duh, warn, as_list_not_str
from ..prob import entropy

class QuantumComputer:
    MATRIX_SLOW = 8
    MATRIX_BREAK = 12
    ENTROPY_EPS = 1e-12
    FILTER_EPS = 1e-12  # filter out small eigenvalues and zero operators
    KEEP_VECTOR = True
    FILTER0 = True  # filter out zero operators

    """
    Simulate a quantum computer! Simulate state vectors or density matrices,
    optionally tracking of the effective quantum channel.

    Additional features:
    - Allows to dynamically add and remove qubits from the system
    - Collapse and decoherent measurements
    - Shortcuts for common channels (like Pauli rotations, noise models, quantum fourier transform)
    - Channel compression via Choi matrix
    - Calculates quantum information metrics (entropy, purity, mutual information)
    - Supports correlation analysis and Schmidt decomposition

    Example:
        # Create a Bell state and track its evolution through a phase estimation circuit
        state_register, energy_register = range(2), range(2, 5)
        qc = QC(np.array([1,0,0,1])/np.sqrt(2), track_operators=True)
        qc.add(energy_register)  # Add 3 ancilla qubits
        qc.pe(unitary, state_register, energy_register)
        qc.measure(energy_register, collapse=False)
        print(qc[0,1])  # look at the state register reduced density matrix
        operators = qc.get_operators()  # Look at the effective channel

    Parameters:
        qubits (list): List of qubits to be allocated.
        state (array): Initial state of the system (default: |0>).
        track_operators (bool): Whether to track the effective quantum channel. Default is 'auto'.
        check (int): Check level for input validation.
    """
    def __init__(self, qubits=None, state=None, track_operators='auto', check=2):
        self._track_operators = track_operators
        self.check_level = check

        self.clear()

        if state is None and qubits is None:
            return
        if state is None and qubits is not None and not is_int(qubits) and is_state(qubits, print_errors=False, check=self.check_level):
            state = qubits
            qubits = count_qubits(state)
        if is_int(qubits):
            qubits = range(qubits)

        qubits = self._check_qubit_arguments(qubits, True)[1]
        self._alloc_qubits(qubits, state=state, track_in_operators=False)

    @property
    def n(self):
        return len(self._qubits)

    @property
    def track_operators(self):
        if isinstance(self._track_operators, bool):
            return self._track_operators
        # track until it gets too large
        if self.n > self.MATRIX_SLOW:
            self._track_operators = False
            return False
        return True

    def clear(self):
        self._state = np.array([1.])
        self._qubits = []
        self._original_order = []
        self.reset_operators()
        return self

    def copy(self):
        qc = QuantumComputer()
        qc._state = self._state.copy()
        qc._qubits = self._qubits.copy()
        qc._added_qubits = self._added_qubits.copy()
        qc._original_order = self._original_order.copy()
        qc._track_operators = self._track_operators
        qc.check_level = self.check_level
        if self.track_operators:
            qc._operators = [o.copy() for o in self._operators]
        return qc

    def __call__(self, operators, qubits='all'):
        qubits, to_alloc = self._check_qubit_arguments(qubits, True)
        self._alloc_qubits(to_alloc)
        operators = assert_kraus(operators, check=0)  # just basic checks

        # if it's a 1-qubit channel and multiple qubits are given, apply it to all qubits
        if operators[0].shape == (2,2) and len(qubits) > 1:
            for q in qubits:
                self(operators, q)
            return self
        operators = assert_kraus(operators, n=(None, len(qubits)), check=self.check_level)

        # non-unitary operators require density matrix representation
        if not is_unitary_channel(operators, check=0):
            self.to_dm()

        self._reorder(qubits, reshape=True)  # required by both update and apply
        # multiply each operator with the new operators
        if self.track_operators:
            self._update_operators(operators)

        # apply operators to state
        self._apply_operators(operators, len(qubits))
        return self

    def _update_operators(self, operators):
        self._operators = combine_channels(operators, self._operators, filter0=self.FILTER0, tol=self.FILTER_EPS, check=0)

    def _apply_operators(self, operators, q):
        self._state = apply_channel(operators, self._state, q != self.n, check=0)

    def get_state(self, qubits='all', obs=None):
        return self._get("_state", qubits, obs)

    def get_unitary(self):
        if not self.track_operators:
            raise ValueError("Operator tracking is disabled")
        self._reorder(self._original_order, reshape=False)
        if not is_unitary_channel(self._operators, check=0):
            raise ValueError("Current channel is non-unitary")
        return self._operators[0].copy()

    def get_operators(self):
        if not self.track_operators:
            raise ValueError("Operator tracking is disabled")
        return self._get("_operators", self._original_order, None)

    def get_qubits(self):
        return self._qubits.copy()

    def is_unitary(self):
        if not self.track_operators:
            raise ValueError("Operator tracking is disabled")
        return is_unitary_channel(self._operators, check=0)

    def _get(self, prop, qubits, obs):
        with self.observable(obs, qubits) as qubits:
            self._reorder(qubits, reshape=False)
            a = getattr(self, prop)
            if len(qubits) == self.n:
                return a.copy()
            return partial_trace(a, range(len(qubits)), reorder=False)

    @contextmanager
    def observable(self, obs=None, qubits='all', return_energies=False):
        qubits = self._check_qubit_arguments(qubits, False)
        if obs is not None:
            obs = self.parse_hermitian(obs, len(qubits), check=self.check_level)
            # if obs is diagonal, use identity as basis (convention clash: computational basis ordering breaks order by ascending eigenvalues)
            diagonal = is_diag(obs)
            if diagonal:
                D = np.diag(obs)
            else:
                D, U = eigh(obs)
                self(U.T.conj(), qubits)  # basis change: Tr(rho @ H) = Tr((U^dagger @ rho @ U) @ D)
        else:
            D = None
        try:
            if return_energies:
                yield qubits, D
            else:
                yield qubits
        finally:
            if obs is not None and not diagonal:
                self(U, qubits)  # back to standard basis

    def probs(self, qubits='all', obs=None):
        with self.observable(obs, qubits) as qubits:
            return self._probs(qubits)

    def probs_pp(self, qubits='all', obs=None, filter_eps=FILTER_EPS, precision=7):
        probs = self.probs(qubits, obs)
        print("Prob       State")
        print("-"*25)
        for i, p in enumerate(probs):
            if p > filter_eps:
                print(f"{p:.{precision}f}  {binstr_from_int(i, len(qubits))}")

    def _probs(self, qubits='all'):
        if self.is_matrix_mode():
            state = self.get_state(qubits)  # calls _reorder(reshape=False)
            return np.diag(state).real  # computational basis
        else:
            self._reorder(qubits, reshape=True)
            probs = np.abs(self._state)**2
            if len(probs.shape) == 1:  # all qubits
                return probs
            return np.sum(probs, axis=1)

    def measure(self, qubits='all', collapse=True, obs=None, return_as='binstr'):
        with self.observable(obs, qubits, return_energies=True) as (qubits, energies):
            if collapse:
                if self._track_operators == True:
                    raise ValueError("Collapse is incompatible with Kraus operators.")
                self.reset_operators()
            probs = self._probs(qubits)
            q = len(qubits)
            if self.is_matrix_mode():
                if collapse:
                    outcome = choice(range(2**q), p=probs)

                    # P = np.kron(dm(outcome, n=q), I_(self.n-q))  # projector
                    # self._state = P @ self._state @ P.conj().T / probs[outcome]
                    mask = np.zeros_like(self._state, dtype=bool)
                    idcs = slice(outcome*2**(self.n-q), (outcome+1)*2**(self.n-q))
                    mask[idcs, idcs] = True
                    self._state[~mask] = 0
                    self._state /= probs[outcome]
                else:
                    # partial measurement of density matrix without "looking" -> decoherence
                    # new_state = np.zeros_like(self._state)
                    # for i, p in enumerate(probs):
                    #     Pi = np.kron(dm(i, n=q), I_(self.n-q))
                    #     new_state += Pi @ self._state @ Pi.conj().T  # *p for weighing and /p for normalization cancel out
                    # self._state = new_state

                    # above is equivalent to throwing away all but the block diagonal elements
                    if q == self.n:
                        self._state = np.diag(np.diag(self._state))
                    else:
                        mask = np.zeros_like(self._state, dtype=bool)
                        for i in range(2**q):
                            idcs = slice(i*2**(self.n-q), (i+1)*2**(self.n-q))
                            mask[idcs, idcs] = True
                        self._state[~mask] = 0

                    if self.track_operators:
                        ops = [dm(i, n=q) for i in range(2**q)]
                        self._reorder(qubits, reshape=True)
                        self._update_operators(ops)
                    return self
            else:
                if collapse:
                    # play God
                    outcome = np.random.choice(2**q, p=probs)
                    # collapse
                    keep = self._state[outcome]
                    self._state = np.zeros_like(self._state)
                    self._state[outcome] = normalize(keep)  # may be 1 or vector
                else:
                    if self.KEEP_VECTOR and entropy(probs) < self.ENTROPY_EPS:  # deterministic outcome implies no entanglement, but loss of information can also happen with the "measurement device" (even if there is no entanglement)
                        warn('Outcome is deterministic -> no decoherence')
                        return self
                    if self.n > self.MATRIX_BREAK:
                        warn("collapse=False for large n. Try using vector collapse (collapse=True) instead of decoherence.")
                    # decohere as as density matrix
                    self.to_dm()
                    return self.measure(qubits, collapse=collapse, obs=obs)

        if return_as == 'energy' and energies is not None:
            return energies[outcome]
        elif return_as == 'energy':
            warn("No observable provided for return_as_energy=True. Returning as outcome index instead.")
        elif return_as == 'binstr':
            return binstr_from_int(outcome, len(qubits))
        return outcome

    def reset(self, qubits='all', collapse=True):
        return self.init(0, qubits, collapse=collapse)

    def resetv(self, qubits=None):
        """
        Special reset for state vector collapse of existing qubits. For this case, this is ~2x faster than (but equivalent to) the general reset method.
        Returns the outcome (standard basis) as binary string.
        """
        if self.is_matrix_mode():
            raise ValueError("Special reset not available for matrix mode")

        if self._track_operators == True:
            warn("Reset is incompatible with operator tracking. Resetting operators.")
        self.reset_operators()
        q = len(qubits)
        new_state = ket(0, n=q)
        if q == self.n:
            self._state = new_state
            self._qubits = qubits
            return self
        probs = self._probs(qubits)  # also moves qubits to the front and reshapes
        outcome = np.random.choice(2**q, p=probs)
        keep = self._state[outcome] / sqrt(probs[outcome])
        self._state = np.zeros_like(self._state)
        self._state[outcome] = keep
        return binstr_from_int(outcome, q)

    def init(self, state, qubits='all', collapse=True):
        qubits, to_alloc = self._check_qubit_arguments(qubits, True)
        original_order = self._original_order + to_alloc  # we want `to_alloc` at the end, but `qubits` may not have them at the end

        # trace out already allocated qubits to be re-initialized
        if qubits != to_alloc:  # avoid empty removal (_check_qubit_arguments guarantees same order)
            assert not any(q in self._added_qubits for q in to_alloc), f"WTF-Error: Qubits to be allocated {to_alloc} are already in added_qubits {self._added_qubits}"
            added_qubits_removed = [q for q in qubits if q in self._added_qubits]
            self.remove([q for q in qubits if q not in to_alloc], collapse=collapse)
            self._added_qubits += added_qubits_removed
            to_alloc = qubits  # (re-)allocate all qubits to be initialized
        elif collapse and self._track_operators == True:
            warn(f"The initialized state of qubits {qubits} is not being tracked. Consider using `collapse=False` if you want to include the new state in the operators.")

        # assert to_alloc == qubits
        self._alloc_qubits(to_alloc, state=state, track_in_operators=not collapse)
        assert sorted(self._qubits) == sorted(original_order), f"Invalid qubit bookkeeping: {self._qubits} != {original_order}"  # sanity check
        # restore original order
        self._original_order = original_order
        return self

    def random(self, n=None):
        n = n or self.n
        assert n, 'No qubits have been allocated yet'
        if self.is_matrix_mode():
            self.init(random_dm(n))
        else:
            self.init(random_ket(n))
        return self

    def add(self, qubits, state=None, track_in_operators=False):
        qubits, to_alloc = self._check_qubit_arguments(qubits, True)
        self._alloc_qubits(to_alloc, state=state, track_in_operators=track_in_operators)
        return self

    def remove(self, qubits, collapse=False, obs=None):
        if collapse and self._track_operators == True:
            raise ValueError("Collapse is incompatible with Kraus operators.")

        with self.observable(obs, qubits) as qubits:
            if len(qubits) == self.n:
                return self.clear()

            if not collapse:
                retain = list(range(len(qubits), self.n))
            if self.is_matrix_mode():
                if collapse:
                    q = len(qubits)
                    probs = self._probs(qubits)
                    outcome = choice(2**q, p=probs)
                    idcs = slice(outcome*2**(self.n-q), (outcome+1)*2**(self.n-q))
                    new_state = self._state[idcs, idcs] / probs[outcome]
                else:
                    self._reorder(qubits, reshape=False)
                    new_state = partial_trace(self._state, retain, reorder=False)
            else:
                if collapse or (self.KEEP_VECTOR and self.entanglement_entropy(qubits) < self.ENTROPY_EPS):
                    # if `qubits` are separable, just remove them
                    probs = self._probs(qubits)  # also moves qubits to the front and reshapes
                    outcome = choice(2**len(qubits), p=probs)
                    new_state = normalize(self._state[outcome])
                    if len(qubits) == self.n - 1:
                        new_state = new_state.reshape([2])
                else:
                    # otherwise, we need to decohere
                    if len(retain) > self.MATRIX_BREAK:
                        warn("Decoherence from state vector for large n. Try using vector collapse (collapse=True) instead of decoherence.")
                        # return self.remove(qubits, collapse=True)
                    # self._reorder(qubits, reshape=False)  # self.entanglement_entropy() already does this
                    new_state = partial_trace(self._state, retain, reorder=False)

        # update operators (non-collapse)
        if self.track_operators and not collapse:
            # construct Kraus operators of the partial trace
            ops = removal_channel(len(qubits))
            self._reorder(qubits, reshape=True)
            self._update_operators(ops)
            # sync operator shaping with new_state (reshape=False, new_state is output space)
            self._operators = [o.reshape(new_state.shape[0], -1) for o in self._operators]
        # update the state after updating the operators, so we can still reorder them simultaneously
        self._state = new_state

        # remove qubits from bookkeeping
        self._qubits = [q for q in self._qubits if q not in qubits]
        self._original_order = [q for q in self._original_order if q not in qubits]
        for q in qubits:
            if q in self._added_qubits:
                self._added_qubits.remove(q)
        return self

    def rename(self, qubit_name_dict):
        for q, name in qubit_name_dict.items():
            assert q in self._qubits, f"Qubit {q} not allocated"
            self._qubits[self._qubits.index(q)] = name
            self._original_order[self._original_order.index(q)] = name
        return self

    def reorder(self, new_order='original'):
        if new_order == 'original':
            new_order = self._original_order
        new_order = self._check_qubit_arguments(new_order, False)
        self._original_order = new_order
        self._reorder(new_order, reshape=False)  # may be unnecessary here, since any next call takes care of reordering as necessary
        return self

    def plot(self, show_qubits='all', obs=None, **kw_args):
        state = self.get_state(show_qubits, obs)
        if len(state.shape) == 2:
            return imshow(state, **kw_args)
        return plotQ(state, **kw_args)

    def plotU(self, **imshow_args):
        U = self.get_unitary()
        return imshow(U, **imshow_args)

    def _check_qubit_arguments(self, qubits, allow_new):
        if isinstance(qubits, slice):
            qubits = self._qubits[qubits]
        elif qubits == 'all':
            qubits = self._original_order
        qubits = as_list_not_str(qubits)
        to_alloc = []
        for q in qubits:
            if q not in self._qubits:
                if allow_new:
                    to_alloc.append(q)
                else:
                    raise ValueError(f"Invalid qubit: {q}")
        assert len(set(qubits)) == len(qubits), f"Qubits should not contain a qubit multiple times, but was {qubits}"
        if allow_new:
            return qubits, to_alloc
        return qubits

    def _alloc_qubits(self, new_qubits, state=None, track_in_operators=False):
        if not new_qubits:
            return
        for q in new_qubits:
            assert q not in self._qubits, f"Qubit {q} already allocated"
        q = len(new_qubits)
        if self.n > 0 and self.track_operators:
            RAM_required = (2**(self.n + q))**2*16*2*max(1, len(self._operators))
            if RAM_required > psutil.virtual_memory().available:
                warn(f"Insufficient RAM! {max(1, len(self._operators))} ({self.n + q}-qubit operators would require {duh(RAM_required)})", stacklevel=3)
        else:
            if self.is_matrix_mode():  # False if self.n == 0
                RAM_required = (2**(self.n + q))**2*16*2
            else:
                RAM_required = (2**(self.n + q))*16*2
            if RAM_required > psutil.virtual_memory().available:
                warn(f"Insufficient RAM! ({self.n + q}-qubit state would require {duh(RAM_required)})", stacklevel=3)

        self._reorder(self._qubits, reshape=False)

        # prepare new state
        if state is None:
            if self.is_matrix_mode():
                state = dm(0, n=q)
            else:
                state = ket(0, n=q)
        else:
            if (isinstance(state, str) and (state == 'random_dm' or state == 'random_mixed')) \
                    or (hasattr(state, 'shape') and len(state.shape) == 2):
                state = dm(state, n=len(new_qubits), check=self.check_level)
                if self.n > 0 and not self.is_matrix_mode():
                    self.to_dm()  # switch to matrix mode
            else:
                if isinstance(state, str) and state == 'random_pure':
                    state = 'random'
                if self.is_matrix_mode():
                    state = dm(state, n=len(new_qubits), check=self.check_level)
                else:
                    state = ket(state, n=len(new_qubits), check=self.check_level)

        self._state = np.kron(self._state, state)

        self._qubits += new_qubits
        self._original_order += new_qubits
        if self.track_operators:
            if track_in_operators:
                ops = extension_channel(state, n=q, check=0)  # state already checked above
                self._operators = [np.kron(oi, oj) for oi in self._operators for oj in ops]  # extend output space, but not input space
                self._added_qubits += new_qubits
            else:
                self._operators = [np.kron(o, I_(q)) for o in self._operators]  # extend both output *and* input space
        else:
            self._operators = []  # if tracking is 'auto', but self.n got too large

    def reset_operators(self):
        if self.track_operators:
            self._operators = [I_(self.n, dtype=complex)]
        else:
            self._operators = []
        self._added_qubits = []

    def _reorder(self, new_order, reshape):
        correct_order = self._qubits[:len(new_order)] == new_order
        if reshape:
            if correct_order and self._state.shape[0] == 2**len(new_order):
                return
        elif correct_order and self._state.shape[0] == 2**self.n:
            return

        assert all(q in self._qubits for q in new_order), f"Invalid qubit order: {new_order} not all in {self._qubits}"
        new_order_all = new_order + [q for q in self._qubits if q not in new_order]
        axes_new = [self._qubits.index(q) for q in new_order_all]
        n_front = len(new_order)
        n = self.n

        def _reorder(a, axes_new, is_matrix):
            if not is_matrix:
                a = a.reshape([2]*n)
                a = a.transpose(axes_new)
                if reshape and n_front < n:
                    a = a.reshape(2**n_front, 2**(n - n_front))
                else:
                    a = a.reshape(2**n)
            else:
                if a.ndim in (3,4):
                    n_in  = count_qubits(prod(a.shape[2:]))
                    # n_out = count_qubits(prod(a.shape[:2]))
                else:
                    n_in  = count_qubits(a.shape[1])
                    # n_out = count_qubits(a.shape[0])
                # assert n == n_out, f"Invalid number of qubits in input space: {n} != {n_out}"
                n_total = n + n_in
                a = a.reshape([2]*n_total)  # order is qubits_out + qubits_in
                if n == n_in:
                    axes_new = axes_new + [i + n for i in axes_new]
                    a = a.transpose(axes_new)
                    if reshape and n_front < n:
                        a = a.reshape([2**n_front, 2**(n - n_front)]*2)
                    else:
                        a = a.reshape([2**n]*2)
                elif n < n_in:
                    n_removed = n_in - n  # number of qubits this operator removes
                    in_axes_new  = [i + n_removed + n for i in axes_new] + list(range(n, n+n_removed))  # input space has additional qubits in beginning that are not in `axes_new`
                    axes_new = axes_new + in_axes_new  # avoid in-place modification
                    a = a.transpose(axes_new)
                    if reshape and n_front < n:
                        a = a.reshape(2**n_front, 2**(n - n_front), 2**n_front, 2**(n + n_removed - n_front))
                    else:
                        a = a.reshape(2**n, 2**(n + n_removed))
                else:  # n_out > n_in
                    n_added = n - n_in
                    assert n_added == len(self._added_qubits), f"Invalid number of qubits in input space: {n_in} != {n - len(self._added_qubits)}"
                    axes_new_in, skipped = [], 0
                    for i, idx in enumerate(axes_new):
                        if self._qubits[idx] in self._added_qubits:
                            skipped += 1
                            continue
                        axes_new_in.append(n + i - skipped)
                    axes_new = axes_new + axes_new_in  # combine output space and filtered input space indices
                    a = a.transpose(axes_new)
                    if reshape and n_front < n:
                        a = a.reshape(2**n_front, 2**(n - n_front), -1)  # the requested qubits may not be in the input space
                    else:
                        a = a.reshape(2**n, 2**(n - n_added))
            return a

        if self.is_matrix_mode():
            self._state = _reorder(self._state, axes_new, True)
        else:
            self._state = _reorder(self._state, axes_new, False)
        if self.track_operators:
            for i, o in enumerate(self._operators):
                # print("reorder operator", i, o.shape, axes_new)
                self._operators[i] = _reorder(o, axes_new, True)

        self._qubits = new_order_all  # update index dictionary with new locations
        self._added_qubits = [q for q in new_order_all if q in self._added_qubits]  # sort the added qubits to the new order
        assert all(q in new_order_all for q in self._added_qubits), f"Added qubits not in new order: {self._added_qubits} not all in {new_order_all}"

    def decohere(self, qubits='all', obs=None):
        return self.measure(qubits, collapse=False, obs=obs)

    def is_matrix_mode(self):
        if self._state.shape[0] == 2**self.n:
            if len(self._state.shape) == 1:
                return False
            return True
        else:  # (q x (n-q)) form
            assert self._state.shape[0] < 2**self.n, f"Invalid state shape: {self._state.shape} for {self.n} qubits {self._qubits}"
            if len(self._state.shape) == 2:
                return False
            return True

    def is_pure(self, qubits='all'):
        rho = self.get_state(qubits)
        if len(rho.shape) == 1:
            return True
        return np.isclose(self.purity(), 1)

    def rank(self, qubits='all', obs=None):
        qubits = self._check_qubit_arguments(qubits, False)
        if len(qubits) == self.n:
            if self.is_matrix_mode():
                state = self.get_state(qubits, obs)
                return np.linalg.matrix_rank(state)
            return 1
        return self.schmidt_number(qubits, obs)  # faster than matrix_rank

    def ensemble(self, obs=None, filter_eps=FILTER_EPS):
        """
        Returns a minimal ensemble of orthnormal kets.
        """
        self._reorder(self._original_order, reshape=False)
        with self.observable(obs):
            return ensemble_from_state(self._state, filter_eps=filter_eps, check=0)

    def ensemble_pp(self, obs=None, filter_eps=FILTER_EPS):
        probs, kets = self.ensemble(obs, filter_eps)
        print(f"Prob      State")
        print("-"*25)
        for p, ket in zip(probs, kets):
            print(f"{p:.6f}  {unket(ket)}")

    def purify(self, sample=False, obs=None):
        """
        Convert density matrix to a state vector representation by purification.
        `sample=True` will sample from the ensemble, otherwise a deterministic purification is performed.
        """
        if not self.is_matrix_mode():
            warn("State is already a vector")
            return self

        with self.observable(obs):
            probs, kets = self.ensemble()
            if sample or len(probs) == 1:
                outcome = choice(len(probs), p=normalize(probs, p=1))
                new_state = kets[outcome]
                n_ancillas = 0
            else:
                # construct purification
                n_ancillas = int(np.ceil(np.log2(len(probs))))
                ancilla_basis = I_(n_ancillas)[:len(probs)]
                pkets = np.sqrt(probs)[:, None] * kets
                # new_state = sum(np.kron(k, a) for k, a in zip(pkets, ancilla_basis))
                # new_state = np.einsum('ia,ib->ab', pkets, ancilla_basis).reshape(-1)  # even slower!
                new_state = np.tensordot(pkets, ancilla_basis, axes=(0, 0)).reshape(-1)

            # find n_ancillas integers that are not in self._qubits
            ancillas = []
            i = self.n
            while len(ancillas) < n_ancillas:
                while i in self._qubits or i in ancillas:
                    i += 1
                ancillas.append(i)

            # initialize the new purified state
            self.init(new_state, self._qubits + ancillas)
        return self

    def to_dm(self):
        """
        Convert state vector to density matrix representation.
        """
        if self.is_matrix_mode():
            # warn("State is already a density matrix")
            return self
        # RAM check
        RAM_required = 2**(2*self.n)*16*2
        if RAM_required > psutil.virtual_memory().available:
            warn(f"Insufficient RAM! ({2*self.n}-qubit density matrix would require {duh(RAM_required)})")

        q = self._state.shape[0]  # keep original reshaping
        self._state = self._state.reshape(2**self.n)
        self._state = np.outer(self._state, self._state.conj())
        if q != 2**self.n:
            nq = 2**self.n // q
            self._state = self._state.reshape(q, nq, q, nq)
        return self

    def to_ket(self, kind='max', return_outcome=False, filter_eps=FILTER_EPS):
        """
        Convert density matrix to state vector representation.
        """
        if not self.is_matrix_mode():
            # warn("State is already a vector")
            return self
        if not is_unitary_channel(self._operators, check=0):
            if self._track_operators == True:
                raise ValueError("State vector representation is not compatible with a non-unitary channel")
            elif self.track_operators:
                warn("State vector representation is not compatible with a non-unitary channel. Resetting operators.")
                self._reset_operators()

        p, kets = self.ensemble(filter_eps=filter_eps)
        if kind == 'max':
            outcome = np.argmax(p)
            self._state = kets[outcome]
        elif kind == 'sample':
            outcome = choice(len(p), p=normalize(p, p=1))
            self._state = kets[outcome]
        else:
            raise ValueError(f"Invalid kind: {kind}. Use 'max' or 'sample'.")
        if return_outcome:
            return outcome
        return self

    def choi_matrix(self):
        """
        Returns the Choi-Jamiołkowski representation of the Kraus operators as a scipy sparse matrix.
        """
        if not self.track_operators:
            raise ValueError("Operator tracking is disabled")

        self._reorder(self._original_order, reshape=False)
        return choi_from_channel(self._operators, n=(self.n, None), check=0)

    def compress_operators(self, filter_eps=FILTER_EPS, force=False):
        if not self.track_operators:
            raise ValueError("Operator tracking is disabled")

        n_out = self.n
        choi_dim = prod(self._operators[0].shape)
        n_in = count_qubits(choi_dim) - n_out
        k = len(self._operators)

        if not force and choi_dim*k > 1e6:
            raise ValueError(f"Calculating {k} singular values of a {choi_dim, choi_dim} Choi matrix for {n_out, n_in} qubits may take a while. Use `force=True` to compute it anyway.")
        choi = self.choi_matrix()
        self._operators = channel_from_choi(choi, n=(n_out, n_in), filter_eps=filter_eps, k=k)
        return self

    def ev(self, obs, qubits='all'):
        qubits = self._check_qubit_arguments(qubits, False)
        state = self.get_state(qubits)
        obs = self.parse_hermitian(obs, len(qubits), check=self.check_level)
        if len(state.shape) == 2:
            ev = trace_product(state, obs)
        else:
            ev = state.conj().T @ obs @ state
        return ev.real

    def std(self, obs, qubits='all', return_ev=False):
        qubits = self._check_qubit_arguments(qubits, False)
        state = self.get_state(qubits)
        obs = self.parse_hermitian(obs, len(qubits), check=self.check_level)
        if len(state.shape) == 2:
            m1 = trace_product(state, obs)
            m2 = trace_product(state, obs @ obs)
        else:
            m1 = state.conj().T @ obs @ state
            m2 = state.conj().T @ obs @ obs @ state
        var = m2 - m1**2
        std = sqrt(var.real)
        if return_ev:
            return std, m1
        return std

    def _gen_pp(self, gen, qubits, sort, head, title, formatter):
        qubits = self._check_qubit_arguments(qubits, False)
        if sort == None:
            skey = None
            word = ""
        elif sort == 'in':
            skey = lambda x: (len(x[0]), x[0])
            word = "First "
        elif sort == 'out':
            skey = lambda x: (len(x[1]), x[1])
            word = "First "
        elif sort == 'asc':
            skey = lambda x: x[2]
            word = "Lowest "
        elif sort == 'desc':
            skey = lambda x: -x[2]
            word = "Highest "
        elif type(sort) == 'function':
            skey = sort
        else:
            raise ValueError(f"Invalid sort parameter: {sort}")

        howmany = "All" if head is None or head + 1 >= 2**len(qubits) - 1 else f"{word}{head}"
        print(f"{howmany} bipartitions:\n" + "-"*(sum(len(str(q)) + 1 for q in qubits) + 3) + f" \t{title}")
        if skey is not None:
            gen = sorted(gen, key=skey)
        for part_in, part_out, *c in gen:
            if head is not None:
                if head == 0:
                    break
                head -= 1
            print(f"{' '.join(str(s) for s in part_in)}  |  {' '.join(str(s) for s in part_out)} \t{formatter(c)}")
        return head is None or head > 0

    def von_neumann_entropy(self, qubits='all', obs=None):
        """
        Calculate the von Neumann entropy of the reduced density matrix of the given qubits.
        """
        state = self.get_state(qubits, obs)
        return von_neumann_entropy(state, check=0)

    def entanglement_entropy(self, qubits, obs=None):
        """
        Calculate the entanglement entropy of the given qubits with respect to the rest of the system.

        Alias for `von_neumann_entropy(qubits)`.
        """
        with self.observable(obs, qubits) as qubits:
            if len(qubits) == self.n:
                raise ValueError("Entanglement entropy requires a bipartition of the qubits")
            return self.von_neumann_entropy(qubits)

    def _entanglement_entropy_gen(self, qubits='all', obs=None):
        """
        Calculate the entanglement entropy of all bipartitions.
        """
        with self.observable(obs, qubits) as qubits:
            if len(qubits) == 1:
                print("No bipartitions can be generated from a single qubit")
                return
            for i, o in bipartitions(qubits, unique=self.is_pure()):
                yield i, o, self.entanglement_entropy(i)

    def entanglement_entropy_pp(self, qubits='all', sort=None, head=300, precision=7, obs=None):
        qubits = self._check_qubit_arguments(qubits, False)
        gen = self._entanglement_entropy_gen(qubits, obs)
        if len(qubits) > 10 and sort is not None or len(qubits) >= 12:
            warn("This may take a while")
        printed_all = self._gen_pp(gen, qubits, sort, head, "Entropy", lambda x: f"{x[0]:.{precision}f}".rstrip('0'))
        # add full state entropy
        if printed_all:
            print(f"\nFull state (i.e. classical) entropy: {self.von_neumann_entropy(qubits):.{precision}f}".rstrip('0'))

    def purity(self, qubits='all'):
        return purity(self.get_state(qubits))

    def _purity_gen(self, qubits='all', obs=None):
        with self.observable(obs, qubits) as qubits:
            for i, o in bipartitions(qubits, unique=self.is_pure()):
                yield i, o, self.purity(i)

    def purity_pp(self, qubits='all', sort='in', head=300, precision=7, obs=None):
        gen = self._purity_gen(qubits, obs)
        printed_all = self._gen_pp(gen, qubits, sort, head, "Purity", lambda x: f'{x[0]:.{precision}f}'.rstrip('0'))
        if printed_all:
            print(f"\nFull state purity: {self.purity():.{precision}f}".rstrip('0'))

    def schmidt_decomposition(self, qubits='all', coeffs_only=False, obs=None, filter_eps=1e-10):
        """
        Schmidt decomposition of a bipartition of the qubits. Returns the Schmidt coefficients and the two sets of basis vectors.
        """
        if self.is_matrix_mode():
            raise ValueError("Schmidt decomposition is not available for density matrices")
        with self.observable(obs, qubits) as qubits:
            if len(qubits) == self.n or len(qubits) == 0:
                raise ValueError("Schmidt decomposition requires a bipartition of the qubits")

            idcs = [self._qubits.index(q) for q in qubits]
            return schmidt_decomposition(self._state.reshape(-1), idcs, coeffs_only, filter_eps, check=0)

    def schmidt_coefficients(self, qubits='all', obs=None, filter_eps=1e-10):
        return self.schmidt_decomposition(qubits, coeffs_only=True, obs=obs, filter_eps=filter_eps)

    def _schmidt_coefficients_gen(self, obs=None, filter_eps=1e-10):
        if self.is_matrix_mode():
            raise ValueError("Schmidt coefficients are not available for density matrices")
        if self.n == 0:
            raise ValueError("No bipartitions can be generated from a single qubit")
        state = self.get_state(obs=obs)
        for i, o in bipartitions(self._qubits, unique=True):
            idcs = [self._qubits.index(q) for q in i]
            coeffs = schmidt_decomposition(state, idcs, True, filter_eps, check=0)
            yield i, o, len(coeffs), coeffs

    def schmidt_coefficients_pp(self, sort=None, head=300, obs=None, filter_eps=1e-5, show_coeffs=True):
        gen = self._schmidt_coefficients_gen(obs, filter_eps)
        if show_coeffs:
            formatter = lambda x: f"{x[0]}: {x[1]}"
            title = "Schmidt number: Schmidt coefficients"
        else:
            formatter = lambda x: f"{x[0]}"
            title = "Schmidt number"
        if self.n >= 14 and sort is not None or self.n >= 20:
            warn("This may take a while")
        self._gen_pp(gen, self._qubits, sort, head, title, formatter)

    def schmidt_number(self, qubits='all', obs=None):
        return len(self.schmidt_coefficients(qubits, obs))

    def correlation(self, qubits_A, qubits_B, obs_A, obs_B):
        """
        Compute the correlation between two subsystems A and B, defined with respective observables obs_A and obs_B.
        """
        qubits_A = self._check_qubit_arguments(qubits_A, False)
        qubits_B = self._check_qubit_arguments(qubits_B, False)
        assert not any(q in qubits_B for q in qubits_A), "Subsystems A and B must be disjoint"
        state = self.get_state(qubits_A + qubits_B)
        obs_A = self.parse_hermitian(obs_A, len(qubits_A), check=self.check_level)
        obs_B = self.parse_hermitian(obs_B, len(qubits_B), check=self.check_level)
        return correlation_quantum(state, obs_A, obs_B, check=0)

    def mutual_information(self, qubits_A, qubits_B=None, obs_A=None, obs_B=None):
        """
        Compute the mutual information between two subsystems A and B, defined as S(A) + S(B) - S(AB). If B is not provided, it is assumed to be the complement of A.
        """
        with self.observable(obs_A, qubits_A) as qubits_A:
            if qubits_B is None:
                qubits_B = [q for q in self._qubits if q not in qubits_A]
            with self.observable(obs_B, qubits_B) as qubits_B:
                assert not any(q in qubits_B for q in qubits_A), "Subsystems A and B must be disjoint"

                state = self.get_state(qubits_A + qubits_B)
                A_idcs = list(range(len(qubits_A)))
                return mutual_information_quantum(state, A_idcs, check=0)

    def noise(self, noise_model=None, qubits='all', p=0.1, obs=None):
        """
        Apply noise to the qubits. See `noise_models.keys()` for available noise models.
        """
        if noise_model is None:
            raise ValueError("No noise model provided. Valid options are: " + ', '.join(noise_models.keys()))
        if noise_model not in noise_models:
            raise ValueError(f"Invalid noise model: {noise_model}. Valid options are: " + ', '.join(noise_models.keys()))
        with self.observable(obs, qubits) as qubits:
            operators = noise_models[noise_model](p)
            return self(operators, qubits)

    def __str__(self):
        try:
            state = self.get_state()
            if self.is_matrix_mode():
                state = '\n' + str(state)
            else:
                state = f"'{unket(state)}'"
        except:
            state = None
        return f"qubits {self._qubits} in state {state}"

    def __repr__(self):
        return self.__str__() + f" at {hex(id(self))}"

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(f"{self.__class__.__name__}(...)")
        else:
            p.text(str(self))

    def __getitem__(self, qubits):
        state = self.get_state(qubits)
        if len(state.shape) == 1:
            state = np.outer(state, state.conj())
        return state

    def __setitem__(self, qubits, state):
        self.init(state, qubits, collapse=False)
        return

    def __delitem__(self, qubits):
        self.remove(qubits)
        return self

    def __neg__(self):
        if not self.is_matrix_mode():
            raise ValueError("Negation is not defined for state vectors")

        self._state = I_(self.n) - self._state.reshape(2**self.n, 2**self.n)
        return self

    def x(self, q):
        return self(X, q)

    def y(self, q):
        return self(Y, q)

    def z(self, q):
       return self(Z, q)

    def h(self, q='all'):
        return self(H_gate, q)

    def t(self, q):
        return self(T_gate, q)

    def t_dg(self, q):
        return self(T_gate.T.conj(), q)

    def s(self, q):
        return self(S, q)

    def cx(self, control, target):
        return self(CX, [control, target])

    def nx(self, control, target):
        return self(NX, [control, target])

    def ccx(self, control1, control2, target):
        return self(Toffoli, [control1, control2, target])

    def c(self, U, control, target, negative=False):
        control = as_list_not_str(control)
        target  = as_list_not_str(target)

        U = self.parse_unitary(U, len(target), check=self.check_level)
        for _ in control:
            U = C_(U, negative=negative)
        return self(U, control + target)

    def cc(self, U, control1, control2, target):
        return self.c(U, [control1, control2], target)

    def swap(self, qubit1, qubit2):
        return self(SWAP, [qubit1, qubit2])

    def cswap(self, control, qubit1, qubit2):
        return self(Fredkin, control, [qubit1, qubit2])

    def rx(self, angle, q):
        return self(Rx(angle), q)

    def ry(self, angle, q):
        return self(Ry(angle), q)

    def rz(self, angle, q):
        return self(Rz(angle), q)

    def rot(self, phi, theta, lam, q):
        return self(Rot(phi, theta, lam), q)

    def qft(self, qubits, inverse=False, do_swaps=True):
        qubits = self._check_qubit_arguments(qubits, False)
        QFT = Fourier_matrix(n=2**len(qubits), swap=not do_swaps)
        if inverse:
            QFT = QFT.T.conj()
        return self(QFT, qubits)

    def iqft(self, qubits, do_swaps=True):
        return self.qft(qubits, inverse=True, do_swaps=do_swaps)

    def pe(self, U, state, energy):
        # 1. Hadamard on energy register
        self.h(energy)

        # 2. Conditioned unitary powers
        U = self.parse_unitary(U, check=self.check_level)
        for j, q in enumerate(energy):
            if j > 0:
                U = U @ U
            self.c(U, q, state)

        # 3. IQFT on energy register
        self.iqft(energy, do_swaps=False)
        return self

    @classmethod
    def from_ensemble(cls, probs, kets):
        state = np.sum(p * np.outer(k, k.conj()) for p, k in zip(probs, kets))
        return cls(state)

    @staticmethod
    def parse_unitary(U, n_qubits=None, check=2):
        if isinstance(U, np.ndarray):
            pass
        elif isinstance(U, list):
            U = np.asarray(U)
        elif isinstance(U, str):
            U = parse_unitary(U)
        elif "scipy" in sys.modules and sp.issparse(U):
            U = U.toarray()
        else:
            try: # qiskit might not be loaded
                U = get_unitary(U)
            except:
                raise ValueError(f"Can't process unitary of type {type(U)}: {U}")
        if check >= 2:
            assert is_unitary(U), f"Matrix is not unitary: {U}"
        if n_qubits is not None:
            n_U = count_qubits(U)
            assert n_U == n_qubits, f"Unitary has {n_U} qubits, but {n_qubits} qubits were provided"
        return U

    @staticmethod
    def parse_hermitian(H, n_qubits=None, check=2):
        if isinstance(H, (np.ndarray, list)):
            H = np.asarray(H)
        elif isinstance(H, str):
            H = parse_hamiltonian(H)
        elif "scipy" in sys.modules and sp.issparse(H):
            H = H.toarray()
        else:
            raise ValueError(f"Can't process observable of type {type(H)}: {H}")
        if check >= 2:
            assert is_hermitian(H), f"Observable is not hermitian: {H}"
        if n_qubits is not None:
            n_obs = count_qubits(H)
            assert n_obs == n_qubits, f"Observable has {n_obs} qubits, but {n_qubits} qubits were provided"
        return H