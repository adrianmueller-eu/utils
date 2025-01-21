import sys, psutil, warnings
from contextlib import contextmanager
import numpy as np
try:
    import scipy.sparse as sp
    from scipy.linalg import eig, eigh, inv
except ImportError:
    from numpy.linalg import eig, eigh, inv

from .constants import *
from .state import partial_trace, ket, dm, unket, count_qubits, random_ket, random_dm, plotQ, is_state, is_dm, as_state
from .hamiltonian import parse_hamiltonian
from .info import von_neumann_entropy, schmidt_decomposition, mutual_information_quantum, correlation_quantum, is_kraus
from .unitary import parse_unitary, get_unitary, Fourier_matrix
from ..mathlib import choice, normalize, binstr_from_int, bipartitions, is_unitary, is_hermitian, is_diag
from ..plot import imshow
from ..utils import is_int, duh
from ..prob import entropy

class QuantumComputer:
    MATRIX_SLOW = 8
    MATRIX_BREAK = 12
    ENTROPY_EPS = 1e-12

    """
    A naive simulation of a quantum computer. Can simulate as state vector or density matrix.
    """
    def __init__(self, qubits=None, state=None, track_unitary='auto', check=2):
        self.track_unitary = track_unitary
        self.check_level = check

        self.clear()

        if state is None and qubits is not None and not is_int(qubits) and is_state(qubits, self.check_level):
            state = qubits
            qubits = count_qubits(state)
        if is_int(qubits):
            qubits = list(range(qubits))
        if state is None:
            self.reset(qubits)
        else:
            self.init(state, qubits)

    @property
    def n(self):
        return len(self.qubits)

    @property
    def _track_unitary(self):
        return self.track_unitary and not (
            self.track_unitary == 'auto' and self.n > self.MATRIX_SLOW
        )

    def clear(self):
        self.state = np.array([1.])
        self.qubits = []
        self.original_order = []
        self._reset_unitary()
        return self

    def copy(self):
        qc = QuantumComputer()
        qc.state = self.state.copy()
        qc.qubits = self.qubits.copy()
        qc.original_order = self.original_order.copy()
        qc.track_unitary = self.track_unitary
        qc.check_level = self.check_level
        if self._track_unitary:
            qc.U = self.U.copy()
        return qc

    def __call__(self, operators, qubits='all'):
        qubits, to_alloc = self._check_qubit_arguments(qubits, True)
        self._alloc_qubits(to_alloc)
        operators = self.parse_channel(operators, len(qubits), check=self.check_level)
        isunitary = len(operators) == 1
        # if it's a 1-qubit channel and multiple qubits are given, apply it to all qubits
        if operators[0].shape == (2,2) and len(qubits) > 1:
            for q in qubits:
                self(operators, q)
            return self

        if not isunitary:
            if self.track_unitary == True:
                warnings.warn("Unitary tracking disabled for multiple operators", stacklevel=2)
            self.track_unitary = False

            if not self.is_matrix_mode():
                self.to_dm()

        # rotate axes of state vector to have the `qubits` first
        if qubits != self.qubits or self.state.shape[0] != 2**len(qubits):
            self._reorder(qubits)

        if isunitary:
            # update unitary if tracked
            if self._track_unitary:
                U = operators[0]
                # (q x q) x (q x q x 2(n-q)) -> q x q x 2(n-q)
                self.U = np.tensordot(U, self.U, axes=1)

        # apply operators
        if self.is_matrix_mode():
            new_state = np.zeros_like(self.state, dtype=complex)
            for K in operators:
                if len(qubits) == self.n:
                    # (q x q) x (q x q) x (q x q) -> q x q
                    new_state += K @ self.state @ K.T.conj()
                else:
                    # (q x q) x (q x (n-q) x q x (n-q)) x (q x q) -> q x (n-q) x q x (n-q)
                    tmp = np.tensordot(K, self.state, axes=1)
                    # (q x (n-q) x q x (n-q)) x (q x q) -> q x (n-q) x (n-q) x q
                    tmp = np.tensordot(tmp, K.T.conj(), axes=(2,0))
                    # q x (n-q) x (n-q) x q -> q x (n-q) x q x (n-q)
                    new_state += tmp.transpose([0, 1, 3, 2])
            self.state = new_state
        else:
            assert len(operators) == 1, "Non-unitary operators can't be applied to state vectors!"
            U = operators[0]
            # (q x q) x (q x (n-q)) -> q x (n-q)  or  (q x q) x q -> q
            self.state = np.tensordot(U, self.state, axes=1)
        return self

    def get_state(self, qubits='all', obs=None):
        return self._get("state", qubits, obs)

    def get_U(self, qubits='all', obs=None):
        if not self._track_unitary:
            raise ValueError("Unitary tracking is disabled")
        return self._get("U", qubits, obs)

    def _get(self, prop, qubits, obs):
        with self.observable(obs, qubits) as qubits:
            self._reorder(qubits, reshape=False)
            a = getattr(self, prop)
            if len(qubits) == self.n:
                return a
            return partial_trace(a, [self.qubits.index(q) for q in qubits], reorder=False)

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

    def probs_pp(self, qubits='all', obs=None, filter_eps=1e-12, precision=7):
        probs = self.probs(qubits, obs)
        print("Prob       State")
        print("-"*25)
        for i, p in enumerate(probs):
            if p > filter_eps:
                print(f"{p:.{precision}f}  {binstr_from_int(i, len(qubits))}")

    def _probs(self, qubits='all'):
        if self.is_matrix_mode():
            state = self.get_state(qubits)
            return np.diag(state).real  # computational basis
        else:
            self._reorder(qubits)
            probs = np.abs(self.state)**2
            if len(probs.shape) == 1:  # all qubits
                return probs
            return np.sum(probs, axis=1)

    def measure(self, qubits='all', collapse=True, obs=None, return_as='binstr'):
        with self.observable(obs, qubits, return_energies=True) as (qubits, energies):
            self._reset_unitary()
            probs = self._probs(qubits)
            q = len(qubits)
            if self.is_matrix_mode():
                self.state = self.state.reshape(2**self.n, 2**self.n)
                if collapse:
                    outcome = choice(range(2**q), p=probs)

                    # P = np.kron(dm(outcome, n=q), I_(self.n-q))  # projector
                    # self.state = P @ self.state @ P.conj().T / probs[outcome]
                    mask = np.zeros_like(self.state, dtype=bool)
                    idcs = slice(outcome*2**(self.n-q), (outcome+1)*2**(self.n-q))
                    mask[idcs, idcs] = True
                    self.state[~mask] = 0
                    self.state /= probs[outcome]
                else:
                    # partial measurement of density matrix without "looking" -> decoherence
                    # new_state = np.zeros_like(self.state)
                    # for i, p in enumerate(probs):
                    #     Pi = np.kron(dm(i, n=q), I_(self.n-q))
                    #     new_state += Pi @ self.state @ Pi.conj().T  # *p for weighing and /p for normalization cancel out
                    # self.state = new_state

                    # above is equivalent to throwing away all but the block diagonal elements
                    if q == self.n:
                        self.state = np.diag(np.diag(self.state))
                    else:
                        mask = np.zeros_like(self.state, dtype=bool)
                        for i in range(2**q):
                            idcs = slice(i*2**(self.n-q), (i+1)*2**(self.n-q))
                            mask[idcs, idcs] = True
                        self.state[~mask] = 0
            else:
                if collapse:
                    # play God
                    outcome = np.random.choice(2**q, p=probs)
                    # collapse
                    keep = self.state[outcome]
                    self.state = np.zeros_like(self.state)
                    self.state[outcome] = normalize(keep)  # may be 1 or vector
                else:
                    if entropy(probs) < self.ENTROPY_EPS:  # deterministic outcome implies no entanglement, but loss of information can also happen with the "measurement device" (even if there is no entanglement)
                        warnings.warn('Outcome is deterministic -> no decoherence', stacklevel=2)
                        return self
                    if self.n > self.MATRIX_BREAK:
                        warnings.warn("collapse=False for large n. Try using vector collapse (collapse=True) instead of decoherence.", stacklevel=2)
                    # decohere as as density matrix
                    self.to_dm()
                    return self.measure(qubits, collapse=collapse, obs=obs)

        if not collapse:
            return self
        if return_as == 'energy' and energies is not None:
            return energies[outcome]
        elif return_as == 'energy':
            print("Warning: No observable provided for return_as_energy=True. Returning as outcome index instead.", stacklevel=2)
        elif return_as == 'binstr':
            return binstr_from_int(outcome, len(qubits))
        return outcome

    def reset(self, qubits=None, collapse=True):
        if qubits is not None:
            self.init(0, qubits, collapse=collapse)
        elif self.n:
            self.init(0, collapse=collapse)
        # else: pass
        return self

    def resetv(self, qubits=None):
        """
        Special reset for state vector collapse of existing qubits. For this case, this is ~2x faster than (but equivalent to) the general reset method.
        """
        if self.is_matrix_mode():
            raise ValueError("Special reset not available for matrix mode")

        q = len(qubits)
        new_state = ket(0, n=q)
        if q == self.n:
            self.state = new_state
            self.qubits = qubits
            return self
        probs = self._probs(qubits)  # also moves qubits to the front and reshapes
        outcome = np.random.choice(2**q, p=probs)
        keep = self.state[outcome] / sqrt(probs[outcome])
        self.state = np.zeros_like(self.state)
        self.state[outcome] = keep
        return self

    def init(self, state, qubits=None, collapse=True):
        if qubits is None:
            if self.n == 0:  # infer `qubits` from `state`
                self.state = as_state(state, check=self.check_level)
                n = count_qubits(self.state)
                self.qubits = list(range(n))
                self.original_order = list(range(n))
                self._reset_unitary()
                return self
            qubits = self.qubits
        else:
            qubits, to_alloc = self._check_qubit_arguments(qubits, True)
            original_order = self.original_order + to_alloc
            if len(qubits) != len(to_alloc):  # avoid empty allocation
                self.remove([q for q in qubits if q not in to_alloc], collapse=collapse)
                to_alloc = qubits

        if (isinstance(state, str) and (state == 'random_dm' or state == 'random_mixed')) \
                or (hasattr(state, 'shape') and len(state.shape) == 2):
            new_state = dm(state, n=len(qubits), check=self.check_level)
            if self.n > 0 and not self.is_matrix_mode():
                # switch to matrix mode
                self.state = dm(self.state, check=0)
        else:
            if isinstance(state, str) and state == 'random_pure':
                state = 'random'
            if self.is_matrix_mode():
                new_state = dm(state, n=len(qubits), check=self.check_level)
            else:
                new_state = ket(state, n=len(qubits))

        if qubits == self.qubits:
            self.state = new_state
        else:
            self._alloc_qubits(to_alloc, state=new_state)
            # restore original order
            self.original_order = original_order

        self._reset_unitary()
        return self

    def random(self, n=None):
        n = n or self.n
        assert n, 'No qubits have been allocated yet'
        if self.is_matrix_mode():
            self.init(random_dm(n))
        else:
            self.init(random_ket(n))
        return self

    def add(self, qubits, state=None):
        if state is None:
            qubits, to_alloc = self._check_qubit_arguments(qubits, True)
            self._alloc_qubits(to_alloc)
        else:
            self.init(state, qubits)
        return self

    def remove(self, qubits, collapse=False, obs=None):
        with self.observable(obs, qubits) as qubits:
            if len(qubits) == self.n:
                return self.clear()
            qubits_indcs = [self.qubits.index(q) for q in qubits]
            retain = [q for q in range(self.n) if q not in qubits_indcs]
            if self.is_matrix_mode():
                if collapse:
                    q = len(qubits)
                    probs = self._probs(qubits)
                    outcome = choice(2**q, p=probs)
                    idcs = slice(outcome*2**(self.n-q), (outcome+1)*2**(self.n-q))
                    self.state = self.state[idcs, idcs] / probs[outcome]
                else:
                    self.state = partial_trace(self.state, retain, reorder=False)
            else:
                if collapse or self.entanglement_entropy(qubits) < self.ENTROPY_EPS:
                    # if no entanglement with others, just remove it
                    probs = self._probs(qubits)  # also moves qubits to the front and reshapes
                    outcome = choice(2**len(qubits), p=probs)
                    self.state = normalize(self.state[outcome])
                    if len(qubits) == self.n - 1:
                        self.state = self.state.reshape([2])
                else:
                    # otherwise, we need to decohere
                    if len(retain) > self.MATRIX_BREAK:
                        warnings.warn("Decoherence from state vector for large n. Try using vector collapse (collapse=True) instead of decoherence.", stacklevel=2)
                        # return self.remove(qubits, collapse=True)
                    self.state = partial_trace(self.state, retain, reorder=False)

        self.qubits = [q for q in self.qubits if q not in qubits]
        self.original_order = [q for q in self.original_order if q not in qubits]
        self._reset_unitary()
        return self

    def rename(self, qubit_name_dict):
        for q, name in qubit_name_dict.items():
            assert q in self.qubits, f"Qubit {q} not allocated"
            self.qubits[self.qubits.index(q)] = name
            self.original_order[self.original_order.index(q)] = name
        return self

    def reorder(self, new_order):
        new_order = self._check_qubit_arguments(new_order, False)
        self.original_order = new_order
        self._reorder(new_order)  # may be unnecessary here
        return self

    def plot(self, show_qubits='all', obs=None, **kw_args):
        state = self.get_state(show_qubits, obs)
        if len(state.shape) == 2:
            return imshow(state, **kw_args)
        return plotQ(state, **kw_args)

    def plotU(self, show_qubits='all', obs=None, **kw_args):
        U = self.get_U(show_qubits, obs)
        return imshow(U, **kw_args)

    def _check_qubit_arguments(self, qubits, allow_new):
        if not allow_new and self.n == 0:
            raise ValueError("No qubits allocated yet")
        if isinstance(qubits, slice):
            qubits = self.qubits[qubits]
        elif isinstance(qubits, str) and qubits == 'all':
            qubits = self.original_order
        elif not isinstance(qubits, (list, tuple, np.ndarray, range)):
            qubits = [qubits]
        qubits = list(qubits)
        assert len(qubits) > 0, "No qubits provided"
        to_alloc = []
        for q in qubits:
            if q not in self.qubits:
                if allow_new:
                    to_alloc.append(q)
                else:
                    raise ValueError(f"Invalid qubit: {q}")
        assert len(set(qubits)) == len(qubits), f"Qubits should not contain a qubit multiple times, but was {qubits}"
        if allow_new:
            return qubits, to_alloc
        return qubits

    def _alloc_qubits(self, new_qubits, state=None):
        if not new_qubits:
            return
        for q in new_qubits:
            assert q not in self.qubits, f"Qubit {q} already allocated"
        q = len(new_qubits)
        if self.n > 0 and self._track_unitary:
            RAM_required = (2**(self.n + q))**2*16*2
            if RAM_required > psutil.virtual_memory().available:
                warnings.warn(f"Insufficient RAM! ({self.n + q}-qubit unitary would require {duh(RAM_required)})", stacklevel=3)
        else:
            if self.is_matrix_mode():  # False if self.n == 0
                RAM_required = (2**(self.n + q))**2*16*2
            else:
                RAM_required = (2**(self.n + q))*16*2
            if RAM_required > psutil.virtual_memory().available:
                warnings.warn(f"Insufficient RAM! ({self.n + q}-qubit state would require {duh(RAM_required)})", stacklevel=3)

        if self.is_matrix_mode():
            state = state if state is not None else dm(0, n=q)
            self.state = np.kron(self.state.reshape(2**self.n, 2**self.n), state)
        else:
            state = state if state is not None else ket(0, n=q)
            self.state = np.kron(self.state.reshape(2**self.n), state)
        self.qubits += new_qubits
        self.original_order += new_qubits
        if self._track_unitary:
            self.U = np.kron(self.U, I_(q))
        elif hasattr(self, 'U'):
            del self.U

    def _reset_unitary(self):
        if self._track_unitary:
            self.U = np.eye(2**self.n, dtype=complex)

    def _reorder(self, new_order, reshape=True):
        new_order_all = new_order + [q for q in self.qubits if q not in new_order]
        axes_new = [self.qubits.index(q) for q in new_order_all]
        self.qubits = new_order_all # update index dictionary with new locations
        n_front = len(new_order)
        n = self.n

        def _reorder(a, axes_new, is_matrix):
            if is_matrix:
                axes_new = axes_new + [i + n for i in axes_new]
            m = 2 if is_matrix else 1
            if any(s > 2 for s in a.shape):
                a = a.reshape([2]*m*n)
            a = a.transpose(axes_new)
            # collect
            if reshape and n_front < n:
                a = a.reshape([2**n_front, 2**(n-n_front)]*m)
            else:
                a = a.reshape([2**n]*m)
            return a

        if self.is_matrix_mode():
            self.state = _reorder(self.state, axes_new, True)
        else:
            self.state = _reorder(self.state, axes_new, False)
        if self._track_unitary:
            self.U = _reorder(self.U, axes_new, True)

    def decohere(self, qubits='all', obs=None):
        return self.measure(qubits, collapse=False, obs=obs)

    def is_matrix_mode(self):
        if self.state.shape[0] == 2**self.n:
            if len(self.state.shape) == 1:
                return False
            return True
        else:  # (q x (n-q)) form
            assert self.state.shape[0] < 2**self.n, f"Invalid state shape: {self.state.shape}"
            if len(self.state.shape) == 2:
                return False
            return True

    def is_pure(self, qubits='all'):
        rho = self.get_state(qubits)
        if len(rho.shape) == 1:
            return True
        return np.isclose(np.trace(rho @ rho), 1)

    def rank(self, qubits='all', obs=None):
        qubits = self._check_qubit_arguments(qubits, False)
        if len(qubits) == self.n:
            if self.is_matrix_mode():
                state = self.get_state(qubits, obs)
                return np.linalg.matrix_rank(state)
            return 1
        return self.schmidt_number(qubits, obs)  # faster than matrix_rank

    def ensemble(self, obs=None, filter_eps=1e-12):
        """
        Returns a minimal ensemble of orthnormal kets.
        """
        self._reorder(self.original_order)

        with self.observable(obs):
            if self.is_matrix_mode():
                if self.n > self.MATRIX_SLOW and is_diag(self.state, eps=0):
                    probs = np.diag(self.state).real
                    kets = I_(self.n)
                    return probs, kets
                probs, kets = eigh(self.state)
                # filter out zero eigenvalues
                mask = probs > filter_eps
                probs = probs[mask]
                kets = kets.T[mask]
                return probs.real, kets
            else:
                return np.array([1.]), np.array([self.state]).T

    def ensemble_pp(self, obs=None, filter_eps=1e-12):
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
            warnings.warn("State is already a vector", stacklevel=2)
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
                ancilla_basis = I_(n_ancillas)
                new_state = np.zeros(2**(self.n + n_ancillas), dtype=complex)
                for i, p in enumerate(probs):
                    new_state += sqrt(p) * np.kron(kets[i], ancilla_basis[i])

            # find n_ancillas integers that are not in self.qubits
            ancillas = []
            i = self.n
            while len(ancillas) < n_ancillas:
                while i in self.qubits or i in ancillas:
                    i += 1
                ancillas.append(i)

            # initialize the new purified state
            self.init(new_state, self.qubits + ancillas)
        return self

    def to_dm(self):
        """
        Convert state vector to density matrix representation.
        """
        if self.is_matrix_mode():
            # warnings.warn("State is already a density matrix", stacklevel=2)
            return self
        # RAM check
        RAM_required = 2**(2*self.n)*16*2
        if RAM_required > psutil.virtual_memory().available:
            warnings.warn(f"Insufficient RAM! ({2*self.n}-qubit density matrix would require {duh(RAM_required)})", stacklevel=2)

        self.state = np.outer(self.state, self.state.conj())
        return self

    def ev(self, obs, qubits='all'):
        qubits = self._check_qubit_arguments(qubits, False)
        state = self.get_state(qubits)
        obs = self.parse_hermitian(obs, len(qubits), check=self.check_level)
        if len(state.shape) == 2:
            ev = np.trace(state @ obs)
        else:
            ev = state.conj().T @ obs @ state
        return ev.real

    def std(self, obs, qubits='all', return_ev=False):
        qubits = self._check_qubit_arguments(qubits, False)
        state = self.get_state(qubits)
        obs = self.parse_hermitian(obs, len(qubits), check=self.check_level)
        if len(state.shape) == 2:
            m1 = np.trace(state @ obs)
            m2 = np.trace(state @ obs @ obs)
        else:
            m1 = state.conj().T @ obs @ state
            m2 = state.conj().T @ obs @ obs @ state
        var = m2 - m1**2
        std = sqrt(var.real)
        if return_ev:
            return std, m1
        return std

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

    def entanglement_entropy_pp(self, qubits='all', sort=None, head=300, precision=7, obs=None):
        qubits = self._check_qubit_arguments(qubits, False)
        gen = self._entanglement_entropy_gen(qubits, obs)
        if len(qubits) > 10 and sort is not None or len(qubits) >= 12:
            warnings.warn("This may take a while", stacklevel=2)
        printed_all = self._gen_pp(gen, qubits, sort, head, "Entropy", lambda x: f"{x[0]:.{precision}f}".rstrip('0'))
        # add full state entropy
        if printed_all:
            print(f"\nFull state (i.e. classical) entropy: {self.von_neumann_entropy(qubits):.{precision}f}".rstrip('0'))

    def schmidt_decomposition(self, qubits='all', coeffs_only=False, obs=None, filter_eps=1e-10):
        """
        Schmidt decomposition of a bipartition of the qubits. Returns the Schmidt coefficients and the two sets of basis vectors.
        """
        if self.is_matrix_mode():
            raise ValueError("Schmidt decomposition is not available for density matrices")
        with self.observable(obs, qubits) as qubits:
            if len(qubits) == self.n or len(qubits) == 0:
                raise ValueError("Schmidt decomposition requires a bipartition of the qubits")

            idcs = [self.qubits.index(q) for q in qubits]
            return schmidt_decomposition(self.state.reshape(-1), idcs, coeffs_only, filter_eps, check=0)

    def schmidt_coefficients(self, qubits='all', obs=None, filter_eps=1e-10):
        return self.schmidt_decomposition(qubits, coeffs_only=True, obs=obs, filter_eps=filter_eps)

    def _schmidt_coefficients_gen(self, obs=None, filter_eps=1e-10):
        if self.is_matrix_mode():
            print("Error: Schmidt coefficients are not available for density matrices")
        if self.n == 0:
            print("Error: No bipartitions can be generated from a single qubit")
            return
        state = self.get_state(obs=obs)
        for i, o in bipartitions(self.qubits, unique=self.is_pure()):
            idcs = [self.qubits.index(q) for q in i]
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
            warnings.warn("This may take a while", stacklevel=2)
        self._gen_pp(gen, self.qubits, sort, head, title, formatter)

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
                qubits_B = [q for q in self.qubits if q not in qubits_A]
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
        return f"qubits {self.qubits} in state {state}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.original_order}) at {hex(id(self))}"

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
        if self.is_matrix_mode():
            self.state = I_(self.n) - self.state.reshape(2**self.n, 2**self.n)
            return self
        else:
            raise ValueError("Negation is not defined for vector states")

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
        if not isinstance(control, (list, np.ndarray)):
            control = [control]
        control = list(control)
        if not isinstance(target, (list, np.ndarray)):
            target = [target]
        target = list(target)
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
        QFT = Fourier_matrix(n=len(qubits), n_is_qubits=True)
        if inverse:
            QFT = QFT.T.conj()  # unitary!
        if not do_swaps:
            # Fourier_matrix already does the swaps, so we need to *undo* them
            n = len(qubits)
            for i in range(n//2):
                self.swap(qubits[i], qubits[n-i-1])
        return self(QFT, qubits)

    def iqft(self, qubits, do_swaps=True):
        return self.qft(qubits, inverse=True, do_swaps=do_swaps)

    def pe(self, U, state, energy):
        # 1. Hadamard on energy register
        self.h(energy)

        # 2. Unitary condition
        U = self.parse_unitary(U, check=self.check_level)
        UD, UU = eig(U)
        # eig unfortunately doesn't necessarily output a unitary if the input is unitary
        # see https://github.com/numpy/numpy/issues/15461
        if self.check_level >= 2 and not is_unitary(UU):
            warnings.warn("Eigendecomposition of the unitary didn't yield unitary transformation matrix. Using inverse instead.", stacklevel=2)
            UU_inv = inv(UU)  # transformation matrix is always invertible
        else:
            UU_inv = UU.conj().T
        for j, q in enumerate(energy):
            U_2j = UU @ (UD[:,None]**(2**j) * UU_inv)
            self.c(U_2j, q, state)

        # 3. IQFT on energy register
        self.iqft(energy, do_swaps=False)
        return self

    @classmethod
    def from_ensemble(cls, probs, kets):
        state = np.sum(p * np.outer(k, k.conj()) for p, k in zip(probs, kets))
        return cls(state)

    @staticmethod
    def parse_channel(operators, n_qubits=None, check=2):
        """
        Ensures `operators` form a valid CPTP map. Returns a list of np.ndarray.
        """
        if isinstance(operators, list):
            operators = np.asarray(operators)
        if len(operators.shape) == 3:
            assert len(operators) > 0, "No operators provided"
            assert is_kraus(operators, check=check), "Operators are not valid Kraus operators"
            return operators
        else:
            # it's probably a unitary!
            U = QuantumComputer.parse_unitary(operators, n_qubits, check)
            return [U]

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
            try:
                # qiskit might not be loaded
                U = get_unitary(U)
            except:
                raise ValueError(f"Can't process unitary of type {type(U)}: {U}")
        if check >= 2:
            assert is_unitary(U), f"Unitary is not unitary: {U}"
        if n_qubits is not None:
            n_U = count_qubits(U)
            assert n_U == n_qubits or n_U == 1, f"Unitary has {n_U} qubits, but {n_qubits} qubits were provided"
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

def evolve(state, U, check=2):
    state = as_state(state, check=check)
    n = count_qubits(state)
    U = QuantumComputer.parse_unitary(U, n, check)
    if len(state.shape) == 1:
        return U @ state
    assert is_dm(state, check=check), "Invalid state: not a density matrix"
    return U @ state @ U.T.conj()