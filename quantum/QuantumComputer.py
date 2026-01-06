import sys, psutil
from contextlib import contextmanager
from math import prod, sqrt
import numpy as np
try:
    import scipy.sparse as sp
except ImportError:
    pass

from .constants import *
from .utils import count_qubits, partial_trace, reorder_qubits
from .state import ket, ket_from_int, dm, unket, plotQ, is_state, ensemble_from_state, is_separable_state
from .hamiltonian import parse_hamiltonian
from .info import *
from .unitary import parse_unitary, get_unitary, Fourier_matrix, get_subunitary, is_separable_unitary
from ..mathlib import choice, normalize, binstr_from_int, bipartitions
from ..mathlib.matrix import normalize, is_unitary, is_hermitian, is_diag, trace_product, eigh, outer, is_isometry, kron_eye
from ..plot import imshow
from ..utils import is_int, duh, warn, as_list_not_str, nbytes, shape_it
from ..prob import entropy

class QuantumComputer:
    MAX_N = 16                # If `track_operators = 'auto'`, number of input + output qubits above which operators are not being tracked anymore.
    MATRIX_WARN = 12          # self.n above which to issue warnings when converting to density matrix due to collapse=False parameter
    FILTER_EPS = 1e-12        # Tolerance for filtering out small eigenvalues and operators
    ENTROPY_EPS = 1e-12       # If measuring a state vector on a subsystem decoherently and `keep_vector == True`, keep the state vector as is if the entropy is below this threshold.
    SPARSITY_THRESHOLD = 0.2  # If superoperator tracking is active, threshold at which to switch between sparse and dense representation of the superoperator
    NOISE_P = 0.1             # Value for noise level if self.noise is called (or scheduler active) using a `noise_models` identifier

    """
    Simulate a quantum computer! Simulate state vectors or density matrices,
    while tracking the effective quantum channel as Kraus operators or superoperator (Choi matrix).

    Additional features:
    - Shortcuts for common channels (like Pauli rotations, noise models, quantum fourier transform)
    - Quantum information metrics (entanglement entropy, purity, mutual information, correlation, schmidt coefficients, etc.)
    - Allows to dynamically add and remove qubits from the system
    - Collapse and decoherent measurements

    Example:
        # Create a Bell state and track its evolution through a phase estimation circuit
        state_register, energy_register = range(2), range(2, 5)
        qc = QC(np.array([1,0,0,1])/np.sqrt(2), track_operators=True)
        qc.add(energy_register)  # Add 3 ancilla qubits
        qc.pe(unitary, state_register, energy_register)
        qc.measure(energy_register, collapse=False)
        print(qc[0,1])  # look at the state register reduced density matrix
        operators = qc.get_operators()  # Look at the effective channel

    Note:
    - State vector mode is not compatible with superoperator mode or non-isometric channels.
    - Operator tracking is not compatible with non-linear operations (e.g. measurement collapse).

    Parameters:
        qubits (None|list|int): List of qubit identifiers for qubits to be allocated.
        state (int|array): Initial state of the system (default: |0>).
        track_operators (bool|str): Whether to track the effective quantum channel. `'auto'` tracks as long as the system is not too large (see `MATRIX_SLOW`).
        as_superoperator (bool): Whether to use superoperator (Choi matrix) representation (True) or Kraus operators (False).
        check (int): Check level for input validation (see `../../check levels.md`)
        noise_schedule (None|str|list|function): `None`, a `noise_models` identifier, a set of Kraus operators, or a callback function like this:
            ```
            def noise_schedule(qubits: list, process_type: str, qc: QuantumComputer):
                \"\"\"
                A sample noise scheduler for the QuantumComputer class.

                Parameters
                ----------
                qubits : list
                    The list of qubits being operated on in the current operation.
                process_type : str
                    The type of process triggering the noise application. Possible values: 'apply' (gate application), 'init' (initialization), 'measure' (before measurement).
                qc : QuantumComputer
                    The quantum computer object, providing access to its state and methods.
                \"\"\"
                P1 = 1e-3
                P2 = 1e-2
                P_MEAS = 2e-2
                if process_type == 'measure':
                    return noise_models['bitflip'](p=P_MEAS)  # return a channel or even just a string (a valid key of `noise_models`) and use qc.NOISE_P for the probability
                if process_type == 'apply':
                    k = len(qubits)
                    p = P1 if k == 1 else p = P2 * (k - 1)
                    qc.noise('depolarizing', qubits, p=p)  # or we use the qc object itself
            ```
        keep_vector (bool): If True, try to keep state vector representation in various operations (e.g. obtaining or plotting the state of a subsystem, qubit removal, decoherent measurement).
                            Requires O(N^3) tests if the state is (and, if tracked, operators are) separable.
        auto_compress (bool): If True and Kraus operators are tracked, compress them after each operation if they exceed the Choi dimension.
        filter0 (bool): Whether to filter out all-zero operators.
        filter_eps (float): Tolerance for filtering out small eigenvalues and operators.
        entropy_eps (float): If measuring a state vector on a subsystem decoherently and `keep_vector == True`, keep the state vector as is if the entropy is below this threshold.
        sparsity_threshold (float): If superoperator tracking is active, threshold at which to switch between sparse and dense representation of the superoperator.
    """
    def __init__(self, qubits=None, state=0, track_operators='auto', as_superoperator=False, check=2,
        keep_vector=True, auto_compress=True, filter0=True,
        filter_eps=FILTER_EPS, entropy_eps=ENTROPY_EPS, sparsity_threshold=SPARSITY_THRESHOLD,
        noise_schedule=None
    ):
        assert isinstance(track_operators, bool) or track_operators == 'auto', f"track_operators must be boolean or 'auto', but was: {track_operators}"

        self._track_operators = track_operators
        if as_superoperator:
            import scipy
            assert scipy.__version__ >= '1.15', "scipy >= 1.15 is required for superoperator mode"
            self._operators = None
        else:
            self._operators = []
        self.check_level = check
        self.noise_schedule = noise_schedule
        self._noise_channel_flag = False

        # constants
        self.KEEP_VECTOR = keep_vector
        self.AUTO_COMPRESS = auto_compress
        self.FILTER_EPS = filter_eps
        self.ENTROPY_EPS = entropy_eps
        self.FILTER0 = filter0
        self.FILTER0_EPS = filter_eps
        self.SPARSITY_THRESHOLD = sparsity_threshold

        self.clear()

        if is_int(state) and state == 0:
            if qubits is None:
                return  # avoid adding a qubit
            if not is_int(qubits) and is_state(qubits, print_errors=False, check=self.check_level):
                state = as_state(qubits, check=self.check_level)
                qubits = count_qubits(state)
        elif qubits is None and is_state(state, print_errors=False, check=self.check_level):
            state = as_state(state, check=self.check_level)
            qubits = count_qubits(state)
        if is_int(qubits):
            qubits = range(qubits)
        elif isinstance(qubits, str):
            qubits = list(qubits)

        qubits = self._check_qubit_arguments(qubits, True)[1]
        self._alloc_qubits(qubits, state=state, track_in_operators=False)

    @property
    def n(self):
        return len(self._qubits)

    @property
    def qubits(self):
        return self._qubits

    @property
    def is_superoperator(self):
        return not isinstance(self._operators, list)

    @property
    def track_operators(self):
        return not self._too_large_to_track_operators(0)

    @track_operators.setter
    def track_operators(self, value):
        if not isinstance(value, bool):
            raise ValueError(f"Argument must be boolean, but was: {value}")
        self._track_operators = value
        if not value:
            self.reset_operators()

    def reset_operators(self):
        self._input_qubits = list(self._qubits)
        if self.track_operators:
            d = 2**self.n
            Id = np.eye(d, dtype=complex)
            if self.is_superoperator:
                self._operators = choi_from_channel(Id, sparse=d > 4, check=0).reshape(d, d, d, d)
            else:
                self._operators = [Id]
        else:
            self._operators = None if self.is_superoperator else []
            self._input_qubits = []
        return self

    def _too_large_to_track_operators(self, added_n):
        if isinstance(self._track_operators, bool):
            return not self._track_operators
        too_large = (self.n + added_n + len(self._input_qubits)) > self.MAX_N
        if too_large:
            self.track_operators = False
        return too_large

    def _no_tracking(self, message):
        if self._track_operators == False:
            return
        elif self._track_operators == True:
            raise ValueError(message)
        elif self.track_operators:
            self.track_operators = False
            # warn(message, stacklevel=3)

    def clear(self):
        if self.is_superoperator:
            self._state = np.array([[1.]])
        else:
            self._state = np.array([1.])
        self._qubits = ()
        self._original_order = []
        self.reset_operators()
        return self

    def copy(self):
        qc = QuantumComputer()
        qc._state = self._state.copy()
        qc._qubits = self._qubits
        qc._original_order = self._original_order.copy()
        qc._track_operators = self._track_operators
        qc.check_level = self.check_level
        if self.is_superoperator:
            qc._operators = self._operators.copy()
        else:
            qc._operators = [o.copy() for o in self._operators]
        return qc

    def __call__(self, operators, qubits='all'):
        qubits, to_alloc = self._check_qubit_arguments(qubits, True)
        self._alloc_qubits(to_alloc)
        operators = assert_kraus(operators, check=0)  # convert into the correct format

        # if it's a 1-qubit channel and multiple qubits are given, apply it to all qubits
        if operators[0].shape == (2,2) and len(qubits) > 1:
            # TODO: benchmark if creating a tensor product is faster (esp for matrix mode / operators)
            for q in qubits:
                self(operators, q)
            return self

        # for convenience, add qubits if there aren't any yet
        if self.n == 0:
            n = count_qubits(operators[0])
            qubits = list(range(n))
            self._alloc_qubits(qubits)

        if self.check_level > 0:
            operators = assert_kraus(operators, n=(None, len(qubits)), check=self.check_level)

        # non-isometric operators require density matrix representation
        if not is_isometric_channel(operators, check=0):
            self.to_dm()

        self._reorder(qubits, separate=True)  # required by both update and apply

        # apply operators to state
        self._apply_operators(operators, qubits)

        # multiply each operator with the new operators
        self._update_operators(operators)

        # apply noise if noise schedule is set
        self._auto_noise(qubits, process_type='apply')

        return self

    def _update_operators(self, operators):
        if not self.track_operators:
            return
        n_out = count_qubits(operators[0].shape[0])
        if not self._too_large_to_track_operators(n_out - self.n):
            if self.is_superoperator:
                sparse = self._use_sparse_superoperator()
                self._operators = update_choi(operators, self._operators, sparse=sparse, check=0)
            else:
                self._operators = combine_channels(operators, self._operators, filter0=self.FILTER0, tol=self.FILTER0_EPS, check=0)
                self._auto_compress()

    def _apply_operators(self, operators, qubits):
        self._state = apply_channel(operators, self._state, len(qubits) != self.n, check=0)
        K0shape = operators[0].shape
        n_in  = len(qubits)
        n_out = n_in if K0shape[0] == K0shape[1] else count_qubits(K0shape[0])
        if n_out > n_in:
            # add new qubits to bookkeeping
            new_qubits = self._get_new_qubits_ids(n_out - n_in)
            self._qubits += tuple(new_qubits)
            self._original_order += new_qubits
        elif n_out < n_in:
            # remove the last n_in - n_out in qubits
            to_remove = qubits[-(n_in - n_out):]
            self._qubits = tuple(q for q in self._qubits if q not in to_remove)
            self._original_order = [q for q in self._original_order if q not in to_remove]

    def _auto_noise(self, qubits, process_type):
        if self.noise_schedule is not None and not self._noise_channel_flag:
            noise_channel = self.noise_schedule
            if callable(noise_channel):
                with self.no_noise():  # prevent recursion
                    noise_channel = noise_channel(list(qubits), process_type, self)
            if noise_channel is not None:
                self.noise(noise_channel, qubits)

    def is_matrix_mode(self):
        if self._state.shape[0] == 2**self.n:
            if len(self._state.shape) == 1:
                return False
            return True
        else:  # (q x (n-q)) form
            if len(self._state.shape) == 2:
                return False
            return True

    def get_state(self, qubits='all', collapse=False, allow_vector=True, obs=None):
        """
        Returns the state for the specified qubits.

        This method moves `qubits` to the *end* of `self._qubits` for internal calculations.
        """
        def _allow_vector():
            return self.KEEP_VECTOR and allow_vector and (not self.track_operators or not self.is_superoperator and self.is_isometric())

        assert isinstance(collapse, bool), f"collapse must be boolean, but was {collapse}"
        with self.observable(obs, qubits) as qubits:
            q = len(qubits)
            if q == 0:
                if _allow_vector():
                    return np.array([1.])
                return np.array([[1.]])
            elif q == self.n:
                self._reorder(qubits, separate=False)
                if self.is_matrix_mode() or _allow_vector():
                    return self._state.copy()
                return outer(self._state)

            to_remove = [q for q in self._qubits if q not in qubits]
            nq = self.n - q
            if collapse:
                probs = self._probs(to_remove)
                outcome = np.random.choice(2**nq, p=probs)
                if self.is_matrix_mode():
                    # order is to_keep + to_remove -> pick the qxq block of outcome
                    reshaped_state = self._state.reshape([2**q, 2**nq]*2)
                    new_state = reshaped_state[:, outcome, :, outcome] / probs[outcome]
                else:
                    # order is to_remove + to_keep -> select the row `outcome`, containing the `to_keep` qubits
                    new_state = self._state[outcome] / sqrt(probs[outcome])
                    if q == 1:
                        new_state = new_state.reshape([2])
                    if not allow_vector:
                        return outer(new_state)
                return new_state

            if q > self.MATRIX_WARN:
                warn("Decoherence from state vector for large n. Try using vector collapse (collapse=True) instead of decoherence.")
            self._reorder(to_remove, separate=False)
            if not self.is_matrix_mode() and _allow_vector() and self.is_separable_state(to_remove) \
                and (not self.track_operators or self.is_unitary() and self.is_unitary(to_remove)):
                # find a non-zero state (-> no diagonalization required)
                self._reorder(to_remove, separate=True)
                new_state = None
                for i in range(2**nq):
                    if not allclose0(self._state[i]):
                        new_state = self._state[i]
                        break
                assert new_state is not None, f"WTF-Error: No non-zero state found in {self._state}"
                new_state = normalize(new_state)
                if q == 1:
                    new_state = new_state.reshape([2])
                return new_state
            # TODO: if _allow_vector(), check purity(rdm) and i.a. convert to ket?
            return partial_trace(self._state, list(range(nq, self.n)), reorder=False)

    def init(self, state, qubits='all', collapse='auto', track_in_operators='auto'):
        assert isinstance(collapse, bool) or collapse == 'auto', f"collapse must be boolean or 'auto', but was {collapse}"
        qubits, to_alloc = self._check_qubit_arguments(qubits, True)
        to_remove = [q for q in qubits if q not in to_alloc]
        if not to_remove:  # all new
            if track_in_operators == 'auto':
                track_in_operators = self.is_matrix_mode()
            return self.add(qubits, state, track_in_operators)

        if collapse == 'auto':
            collapse = not self.is_matrix_mode()
        if track_in_operators == 'auto':
            track_in_operators = not collapse and (self.track_operators or self.n == len(to_remove))

        # trace out already allocated qubits to be re-initialized
        if track_in_operators:
            # bookkeeping
            original_order = self._original_order + to_alloc  # we want `to_alloc` at the end, but `qubits` may not have them at the end

            # if input is dm, switch to matrix mode
            if not self.is_matrix_mode() and is_square(state):
                self.to_dm()

            # remove qubits from state and operators
            self.remove(to_remove, collapse=collapse, obs=None)
            # extend state and operators by all `qubits`
            self._alloc_qubits(qubits, state=state, track_in_operators=True)

            # bookkeeping
            self._original_order = original_order
            assert sorted(self._qubits) == sorted(original_order), f"Invalid qubit bookkeeping: {self._qubits} != {original_order}"  # sanity check
        else:
            to_retain = [q for q in self._qubits if q not in to_remove]
            reduced_state = self.get_state(to_retain, collapse=collapse, allow_vector=True)  # moves `to_retain` to the end in self._qubits
            # extend operators by identity
            self._reorder(to_retain + to_remove, separate=False)  # to_remove to the end of operators
            self._alloc_qubits(to_alloc, track_in_operators=False)  # add to_alloc at the end
            # extend state with new state
            self._state = reduced_state  # guarantees separate=False for state
            self._qubits = tuple(to_retain)  # ._extend_state (.is_matrix_mode) requires self._qubits to be consistent with self._state.shape
            # we need `state` in the order to_remove + to_alloc, to be in sync with operators
            if not isinstance(state, str) and hasattr(state, '__len__'):
                new_order = [qubits.index(q) for q in to_remove] + [qubits.index(q) for q in to_alloc]
                state = reorder_qubits(state, new_order, separate=False)
            self._extend_state(state, len(qubits))
            # order is now:
            self._qubits = tuple(to_retain + to_remove + to_alloc)
            # recover order to_retain + qubits
            if to_remove + to_alloc != qubits:  # they are set-wise the same, but could be in a different order
                self._reorder(to_retain + qubits)  # move to_alloc where it is in `qubits`

        self._auto_noise(qubits, process_type='init')
        return self

    def reset(self, qubits='all', collapse='auto', track_in_operators='auto'):
        return self.init(0, qubits, collapse, track_in_operators)

    def resetv(self, qubits=None):
        """
        Special reset for state vector collapse of existing qubits (with operator tracking deactivated). For this case, this is equivalent to but faster than the general reset method.
        Returns the outcome (standard basis) as binary string.
        """
        if self.is_matrix_mode():
            raise ValueError("Special reset not available for matrix mode")
        self._no_tracking("Resetv is incompatible with operator tracking.")

        qubits = self._check_qubit_arguments(qubits, False)
        q = len(qubits)
        if q == self.n:
            self._state = ket_from_int(0, n=q)
            self._qubits = tuple(qubits)
            return self
        probs = self._probs(qubits)  # also moves qubits to the front and separates them out
        outcome = np.random.choice(2**q, p=probs)
        keep = self._state[outcome] / sqrt(probs[outcome])
        self._state = np.zeros_like(self._state, dtype=self._state.dtype)
        self._state[0] = keep  # same order as self._qubits
        return binstr_from_int(outcome, q)

    def measure(self, qubits='all', collapse=True, obs=None, return_as='binstr'):
        assert isinstance(collapse, bool), f"collapse must be boolean, but was {collapse}"
        with self.observable(obs, qubits, return_energies=True) as (qubits, energies):
            if collapse:
                self._no_tracking("Collapse is incompatible with operator tracking.")

            self._auto_noise(qubits, process_type='measure')

            probs = self._probs(qubits)
            q = len(qubits)
            d_q = 2**q
            if self.is_matrix_mode():
                if collapse:
                    outcome = np.random.choice(range(d_q), p=probs)

                    # P = np.kron(dm(outcome, n=q), I_(self.n-q))  # projector
                    # self._state = P @ self._state @ P.conj().T / probs[outcome]
                    reshaped_state = self._state.reshape([2**(self.n-q), d_q]*2)
                    kept_block = reshaped_state[:, outcome, :, outcome].copy()
                    reshaped_state[:] = 0  # Zero out everything
                    reshaped_state[:, outcome, :, outcome] = kept_block
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
                        # Zero out off-diagonal elements in the measured space using a broadcasted identity matrix
                        reshaped_state = self._state.reshape([2**(self.n-q), d_q]*2)
                        reshaped_state *= np.eye(d_q, dtype=bool).reshape(1, d_q, 1, d_q)
                    if self.track_operators:
                        ops = [dm(i, n=q) for i in range(2**q)]
                        self._reorder(qubits, separate=True)
                        self._update_operators(ops)
                    return self
            else:
                if collapse:
                    # play God
                    outcome = np.random.choice(d_q, p=probs)
                    # collapse
                    keep = self._state[outcome] / sqrt(probs[outcome])
                    self._state = np.zeros_like(self._state)
                    self._state[outcome] = keep  # may be 1 or vector
                else:
                    if self.KEEP_VECTOR and entropy(probs) < self.ENTROPY_EPS:  # deterministic outcome implies no entanglement, but loss of information can also happen with the "measurement device" (even if there is no entanglement)
                        warn('Outcome is deterministic -> no decoherence')
                        return self
                    if self.n > self.MATRIX_WARN:
                        warn("collapse=False for large n. Try using vector collapse (collapse=True) instead of decoherence.")
                    # decohere as as density matrix
                    self.to_dm()
                    return self.measure(qubits, collapse=collapse, obs=obs)

        if return_as == 'energy':
            if energies is None:
                return (-1)**(outcome.bit_count())  # Z observable
            return energies[outcome]
        elif return_as == 'binstr':
            return binstr_from_int(outcome, q)
        return outcome

    def decohere(self, qubits='all', obs=None):
        return self.measure(qubits, collapse=False, obs=obs)

    def sample(self, qubits='all', shots=None, return_as='binstr', obs=None):
        with self.observable(obs, qubits, return_energies=True) as (qubits, energies):
            probs = self.probs(qubits)
            outcomes = np.random.choice(len(probs), shots, p=probs, replace=True)
            if return_as == 'energy':
                if energies is None:
                    if isinstance(outcomes, int):
                        return (-1)**outcomes.bit_count()
                    arr = np.empty_like(outcomes, dtype=int)
                    for idx in shape_it(outcomes):
                        arr[idx] = outcomes[idx].bit_count()
                    return (-1)**arr
                return energies[outcomes]
            elif return_as == 'binstr':
                n = len(qubits)
                if isinstance(outcomes, int):
                    return f"{outcomes:0{n}b}"
                arr = np.empty_like(outcomes, dtype=object)
                for idx in shape_it(outcomes):
                    arr[idx] = f"{outcomes[idx]:0{n}b}"
                return arr
            return outcomes

    def probs(self, qubits='all', obs=None):
        with self.observable(obs, qubits) as qubits:
            return self._probs(qubits)

    def probs_pp(self, qubits='all', obs=None, filter_eps=None, precision=7):
        if filter_eps is None:
            filter_eps = self.FILTER_EPS
        probs = self.probs(qubits, obs)
        print("Prob       State")
        print("-"*25)
        for i, p in enumerate(probs):
            if p > filter_eps:
                print(f"{p:.{precision}f}  {binstr_from_int(i, len(qubits))}")

    def _probs(self, qubits='all'):
        """
        calls _reorder(separate=True) for vector and _reorder(separate=True) on the REMAINING qubits for matrix
        """
        if self.is_matrix_mode():
            state = self.get_state(qubits, collapse=False)
            return np.diag(state).real  # computational basis
        else:
            self._reorder(qubits, separate=True)
            probs = np.abs(self._state)**2
            if len(probs.shape) == 1:  # all qubits
                return probs
            return np.sum(probs, axis=1)

    def add(self, qubits, state=0, track_in_operators=True):
        qubits, to_alloc = self._check_qubit_arguments(qubits, True)
        self._alloc_qubits(qubits, state=state, track_in_operators=track_in_operators)
        return self

    def remove(self, qubits, collapse=False, obs=None):
        assert isinstance(collapse, bool), f"collapse must be boolean, but was {collapse}"
        if collapse:
            self._no_tracking("Collapse is incompatible with operator tracking.")

        with self.observable(obs, qubits) as qubits:
            if len(qubits) == 0:
                return self

            q = len(qubits)
            to_retain = [q for q in self._qubits if q not in qubits]
            new_state = self.get_state(to_retain, collapse=collapse)

            # update operators
            if self.track_operators:
                assert not collapse, "WTF-Error: Collapse is incompatible with operator tracking."
                if new_state.ndim == 1:
                    # unitary was separable
                    U = self.get_unitary(to_retain)  # this needs self._state / self._qubits not updated yet
                    if self.is_superoperator:
                        d = new_state.shape[0]  # 2**len(to_retain)
                        self._operators = choi_from_channel([U]).reshape(d,d,d,d)
                        self._use_sparse_superoperator()  # check to convert to dense
                    else:
                        self._operators = [U]
                    self._input_qubits = [q for q in self._input_qubits if q in to_retain]
                else:
                    # construct Kraus operators of the partial trace
                    ops = removal_channel(q)
                    self._reorder(qubits, separate=True)
                    self._update_operators(ops)
                    # sync operator shaping with new_state (separate=False, new_state is output space)
                    d_out = new_state.shape[0]  # 2**len(to_retain)
                    if self.is_superoperator:
                        d_in = self._operators.shape[-1]  # 2**len(self._input_qubits)
                        self._operators = self._operators.reshape(d_out, d_in, d_out, d_in)
                    else:
                        self._operators = [o.reshape(d_out, -1) for o in self._operators]

            # update the state after updating the operators, so we can still reorder them simultaneously
            self._state = new_state

        # remove qubits from bookkeeping
        self._qubits = tuple(to_retain)
        self._original_order = [q for q in self._original_order if q not in qubits]
        return self

    def rename(self, qubit_name_dict):
        tmp_qubits = list(self._qubits)
        for q, name in qubit_name_dict.items():
            assert q in tmp_qubits, f"Qubit {q} not allocated"
            tmp_qubits[tmp_qubits.index(q)] = name
            self._original_order[self._original_order.index(q)] = name
        self._qubits = tuple(tmp_qubits)
        return self

    def reorder(self, new_order='original'):
        if new_order == 'original':
            new_order = self._original_order
        new_order = self._check_qubit_arguments(new_order, False)
        self._original_order = new_order
        self._reorder(new_order, separate=False)  # may be unnecessary here, since any next call takes care of reordering as necessary
        return self

    def plot(self, show_qubits='all', obs=None, **kw_args):
        state = self.get_state(show_qubits, obs=obs)
        if len(state.shape) == 2:
            return imshow(state, **kw_args)
        return plotQ(state, **kw_args)

    def plotU(self, **imshow_args):
        V = self.get_isometry()
        return imshow(V, **imshow_args)

    def to_dm(self):
        """
        Convert state vector to density matrix representation.
        """
        if self.is_matrix_mode():
            # warn("State is already a density matrix")
            return self
        # RAM check
        RAM_required = 2**(2*self.n)*8*2
        if RAM_required > psutil.virtual_memory().available:
            warn(f"Insufficient RAM! ({2*self.n}-qubit density matrix would require {duh(RAM_required)})")

        q = self._state.shape[0]  # keep original reshaping
        self._state = self._state.reshape(2**self.n)
        self._state = outer(self._state)
        if q != 2**self.n:
            nq = 2**self.n // q
            self._state = self._state.reshape(q, nq, q, nq)
        return self

    def to_ket(self, kind='sample', return_outcome=False, filter_eps=None):
        """
        Convert density matrix to state vector representation.
        """

        if not self.is_matrix_mode():
            # warn("State is already a vector")
            return self
        if self.is_superoperator:
            raise ValueError("State vector mode is not supported for superoperator representation")

        if filter_eps is None:
            filter_eps = self.FILTER_EPS
        if kind == 'purify':
            if return_outcome:
                raise ValueError("return_outcome is not supported for kind='purify'")
            return self.purify()

        if not is_isometric_channel(self._operators, check=0):
            self._no_tracking("State vector representation is not compatible with a non-isometric channel")

        p, kets = self.ensemble(filter_eps=filter_eps)
        if len(p) == 1:
            self._state = kets[0]
            return 0 if return_outcome else None

        self._no_tracking("Non-linear state conversion is not compatible with operator tracking.")
        if kind == 'sample':
            outcome = np.random.choice(len(p), p=normalize(p, p=1))
            self._state = kets[outcome]
        elif kind == 'max':
            outcome = np.argmax(p)
            self._state = kets[outcome]
        else:
            raise ValueError(f"Invalid kind: {kind}. Use 'purify', 'sample', or 'max'.")
        if return_outcome:
            return outcome
        return self

    def get_unitary(self, qubits='all', obs=None):
        qubits = self._check_qubit_arguments(qubits, False)
        try:
            U = self.get_isometry(obs=obs)  # calls separate=False
            isunitary = is_square(U)
        except AssertionError:
            isunitary = False

        if len(qubits) == self.n:
            if not isunitary:
                raise AssertionError("Current channel is non-unitary")
            return U

        if not isunitary:
            raise NotImplementedError("Can't deduce local operation of non-unitary channel")
        try:
            U1 = get_subunitary(U, [self._qubits.index(q) for q in qubits], check=0, check_output=True)
        except AssertionError:
            return AssertionError(f"Unitary is not separable on requested subsystem: {qubits}")
        return U1

    @property
    def U(self):
        assert not self.is_superoperator, "Property U only available for Kraus operator tracking"
        return self.get_unitary()

    def get_isometry(self, obs=None):
        if not self.track_operators:
            raise ValueError("Operator tracking is disabled")
        with self.observable(obs):
            if self.is_superoperator:
                V = channel_from_choi(self.choi_matrix(), dims=(2**self.n, 2**len(self._input_qubits)), k=1)[0]
                if is_isometry(V, kind='right'):
                    return V
            else:
                if is_isometric_channel(self._operators, check=0):
                    self._reorder(self._original_order, separate=False)  # self.choi_matrix() calls this, too
                    return self._operators[0]
            raise AssertionError("Current channel is non-isometric")

    def is_unitary(self, qubits='all', obs=None):
        qubits = self._check_qubit_arguments(qubits, False)
        if not self.is_superoperator and len(qubits) == self.n:
            return is_unitary_channel(self._operators, check=0)

        try:
            U = self.get_unitary(obs=obs)  # calls separate=False
        except AssertionError:
            U = None
        if len(qubits) == self.n:
            return U is not None

        if U is None:
            raise NotImplementedError("Can't deduce local operation of non-unitary channel")
        # if current channel is unitary, check if unitary can be decomposed into unitaries U = U1 \otimes U2
        return is_separable_unitary(U, [self._qubits.index(q) for q in qubits], check=0)

    def is_isometric(self):
        if self.is_superoperator:
            try:
                self.get_isometry()
                return True
            except AssertionError:
                return False
        return is_isometric_channel(self._operators, check=0)

    def is_pure(self, qubits='all'):
        return np.isclose(self.purity(qubits), 1)

    def is_separable_state(self, qubits='all', obs=None):
        """
        Check if the state is separable with respect to the given qubits.
        """
        with self.observable(obs, qubits) as qubits:
            qubits_idcs = [self._qubits.index(q) for q in qubits]
            if self.is_matrix_mode():
                state = self._state.reshape(2**self.n,-1)
            else:
                state = self._state.reshape(2**self.n)
            return is_separable_state(state, qubits_idcs, check=0)

    def is_separable(self, qubits='all', obs=None):
        with self.observable(obs, qubits) as qubits:
            return self.is_separable_state(qubits) and self.is_unitary(qubits)

    def get_operators(self):
        if not self.track_operators:
            raise ValueError("Operator tracking is disabled")
        self._reorder(self._original_order, separate=False)
        if self.is_superoperator:
            return channel_from_choi(self._operators, dims=(2**self.n, 2**len(self._input_qubits)), filter_eps=self.FILTER_EPS)
        return self._operators

    def choi_matrix(self):
        """
        Returns the Choi-Jamiołkowski representation of the Kraus operators.
        """
        if not self.track_operators:
            raise ValueError("Operator tracking is disabled")

        if self.is_superoperator:
            choi_dim = 2**(self.n + len(self._input_qubits))
            return self._operators.reshape(choi_dim, choi_dim)
        self._reorder(self._original_order, separate=False)
        return choi_from_channel(self._operators, filter_eps=self.FILTER_EPS, check=0)

    def _auto_compress(self):
        if not self.AUTO_COMPRESS or not self.track_operators:
            return
        if self.is_superoperator:
            return self.compress_operators()

        k = len(self._operators)
        if k > 4:
            max_k = 2**(2*min(self.n, len(self._input_qubits)))
            if k > max_k:  # guaranteed not to be minimal
                self.compress_operators()

    def to_choi(self):
        if self.is_superoperator:
            warn("Already in superoperator mode")
            return self
        if not self.is_matrix_mode():
            warn("Superoperator requires state in density matrix -> converting state to density matrix")
            self.to_dm()
        self._operators = self.choi_matrix()

    def to_kraus(self):
        if not self.is_superoperator:
            warn("Already in Kraus mode")
            return self
        self._operators = self.get_operators()

    def compress_operators(self, filter_eps=None):
        if not self.track_operators:
            raise ValueError("Operator tracking is disabled")

        if filter_eps is None:
            filter_eps = self.FILTER_EPS

        if self.is_superoperator:
            if sp.issparse(self._operators):
                # filter numbers smaller than filter_eps from sparse array
                self._operators.data[np.abs(self._operators.data) < filter_eps] = 0
                self._operators.eliminate_zeros()
            else:
                self._operators[self._operators < filter_eps] = 0
                self._use_sparse_superoperator()  # check to convert to sparse
        else:
            self._operators = compress_channel(self._operators, filter_eps=filter_eps, check=0)
        return self

    def _use_sparse_superoperator(self):
        if self.is_superoperator and self._operators is not None:
            threshold = self.SPARSITY_THRESHOLD*prod(self._operators.shape)  # smaller -> sparse, larger -> dense
            n_tot = self.n + len(self._input_qubits)
            if sp.issparse(self._operators):
                # print("Is sparse: ", self._operators.nnz, prod(self._operators.shape), f"{self._operators.nnz/prod(self._operators.shape):.2%}")
                if n_tot <= 6 or self._operators.nnz > threshold:
                    self._operators = self._operators.toarray()
                    return False
                return True
            # print("Is dense: ", np.count_nonzero(self._operators), prod(self._operators.shape), f"{np.count_nonzero(self._operators)/prod(self._operators.shape):.2%}")
            if self.n > 6 and np.count_nonzero(self._operators) < threshold:
                self._operators = sp.coo_array(self._operators)
                return True
            return False

    def _get_new_qubits_ids(self, q):
        """ Returns the new qubit ids for the given number of qubits. """
        start_id = max([q for q in self._qubits if isinstance(q, int)], default=-1) + 1
        return [q + start_id for q in range(q)]

    def _check_qubit_arguments(self, qubits, allow_new):
        if qubits == 'all':
            if allow_new:
                return self._original_order, []
            return self._original_order
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

    def _alloc_qubits(self, new_qubits, state=0, track_in_operators=False):
        if not new_qubits:
            if isinstance(state, int) and state == 0:
                return
            state = as_state(state)
            n = count_qubits(state)
            if self.n == 0:
                new_qubits = list(range(n))
            else:
                raise ValueError("State, but no qubits to provided on with qubits already allocated")
        for q in new_qubits:
            assert q not in self._qubits, f"Qubit {q} already allocated"
        q = len(new_qubits)
        if self.n > 0 and self.track_operators:
            if self.is_superoperator:
                RAM_required = (2**(self.n + q + len(self._input_qubits)))**2*8*2
            else:
                RAM_required = (2**(self.n + q + len(self._input_qubits)))*8*2*max(1, len(self._operators))
            if RAM_required > psutil.virtual_memory().available:
                warn(f"Insufficient RAM! {max(1, len(self._operators))} ({self.n + q}-qubit operators would require {duh(RAM_required)})", stacklevel=3)
        else:
            if self.is_matrix_mode():  # False if self.n == 0
                RAM_required = (2**(self.n + q))**2*8*2
            else:
                RAM_required = (2**(self.n + q))*8*2
            if RAM_required > psutil.virtual_memory().available:
                warn(f"Insufficient RAM! ({self.n + q}-qubit state would require {duh(RAM_required)})", stacklevel=3)

        self._reorder([], separate=False)
        self._extend_state(state, q)

        n_added = q if track_in_operators else 2*q
        if not self._too_large_to_track_operators(n_added):
            if track_in_operators:  # only modifies only the output space, even for unseen qubits
                ops = extension_channel(state, n=q, check=0)  # state already checked above
                if self.is_superoperator:
                    d_out, d_in = self._operators.shape[:2]
                    self._operators = self._operators.reshape(1, d_out, d_in, 1, d_out, d_in)
                    self._update_operators(ops)
                else:
                    self._operators = [np.kron(oi, oj) for oi in self._operators for oj in ops]  # extend output space, but not input space
            else:  # modifies both output and input space for unseen qubits
                q_new_in = [q for q in new_qubits if q in self._input_qubits]
                n_q_new_in = len(q_new_in)
                if self.is_superoperator:
                    d_out, d_in = self._operators.shape[:2]
                    # these already exist -> extend only output space
                    if n_q_new_in > 0:
                        d_q_new_in = 2**n_q_new_in
                        # extend output space by repeating choi 2*d_q_new_in times
                        if self._use_sparse_superoperator():
                            choi = self._operators.reshape(d_out*d_in, -1)  # sp.kron wants 2d
                            ones = sp.coo_array(np.ones((d_q_new_in**2, 1)))
                            choi = sp.kron(ones, choi)
                            choi = choi.reshape(d_q_new_in, d_q_new_in, d_out, d_in, d_out, d_in)
                            choi = choi.transpose([2, 0, 3, 4, 1, 5])  # out x q x in x out x q x in
                            choi = choi.reshape(d_out*d_q_new_in, d_in, d_out*d_q_new_in, d_in)
                        else:
                            choi = np.repeat(self._operators, d_q_new_in, axis=0)  # much faster than np.kron
                            choi = np.repeat(self._operators, d_q_new_in, axis=2)
                        self._operators = choi
                    # extend both output *and* input space by the others
                    if q - n_q_new_in > 0:
                        d_q = 2**q
                        if self._use_sparse_superoperator() or q > 2:  # identity channel is very sparse
                            choi = self._operators.reshape(d_out*d_in, -1)  # sp.kron wants them 2D
                            choi = sp.coo_array(choi)
                            id = choi_from_channel([np.eye(d_q)], sparse=True, check=0)  # identity channel
                            choi = sp.kron(choi, id)  # extend both input and output space
                            # move new q output qubits after output space (and before input space)
                            choi = choi.reshape([d_out, d_in, d_q, d_q]*2)
                            choi = choi.transpose([0, 2, 1, 3] + [4, 6, 5, 7])
                            choi = choi.reshape([d_out*d_q, d_in*d_q]*2)
                        else:
                            choi = kron_with_id_channel(self._operators, d_q, dims=(d_out, d_in), back=True)
                        self._operators = choi
                else:
                    if n_q_new_in > 0:
                        d_q_new_in = 2**n_q_new_in
                        self._operators = [np.repeat(o, d_q_new_in, axis=0) for o in self._operators]  # these already exist -> extend only output space
                    if q - n_q_new_in > 0:
                        d_new = 2**(q - n_q_new_in)
                        self._operators = [kron_eye(d_new, o, back=True) for o in self._operators]  # extend both output *and* input space by the others
                self._input_qubits = [q for q in self._input_qubits if q not in new_qubits]  # move q_new_in to the end
                self._input_qubits += q_new_in + [q for q in new_qubits if q not in q_new_in]

        # update bookkeeping
        self._qubits += tuple(new_qubits)
        self._original_order += new_qubits

    def _extend_state(self, new_state, q):
        # prepare new state
        if not self.is_matrix_mode() and is_square(new_state):
            self.to_dm()  # switch to matrix mode
        if self.is_matrix_mode():
            new_state = dm(new_state, n=q, check=self.check_level)
        else:
            new_state = ket(new_state, n=q, check=self.check_level)
        self._state = np.kron(self._state, new_state)

    def _reorder(self, new_order, separate):
        assert all(q in self._qubits for q in new_order), f"Invalid qubits: {new_order}"
        n, q = self.n, len(new_order)
        d_n, d_q = 2**n, 2**q
        correct_order = list(self._qubits[:q]) == new_order

        # shortcut if no reordering / reshaping is needed
        if correct_order and self._state.shape[0] == (d_q if separate else d_n):
            return

        d_nq = 2**(n-q)
        n_in = len(self._input_qubits)
        d_in = 2**n_in

        if separate and q < n:
            def get_shape(kind):
                match kind:
                    case 'ket':  return [d_q, d_nq]
                    case 'dm':   return [d_q, d_nq]*2
                    case 'op':   return [d_q, d_nq, d_in]
                    case 'choi': return [d_q, d_nq, d_in]*2
        else:
            def get_shape(kind):
                match kind:
                    case 'ket':  return [d_n]
                    case 'dm':   return [d_n]*2
                    case 'op':   return [d_n, d_in]
                    case 'choi': return [d_n, d_in]*2

        if not correct_order:
            new_order_all = new_order + [q for q in self._qubits if q not in new_order]
            axes_new = [self._qubits.index(q) for q in new_order_all]
            if self.track_operators:
                # generally, the input space may...
                # - ... not have qubits that were added to the output space
                axes_new_in = [n + self._input_qubits.index(q) for q in new_order_all if q in self._input_qubits]
                # - ... have qubits that are not present anymore in the output space (order unchanged)
                axes_new_in = axes_new_in + [i for i in range(n,n+n_in) if i not in axes_new_in]
                axes_new_ = axes_new + axes_new_in  # order is qubits_out + qubits_in
                n_tot = n + n_in
            _transpose = lambda a, n_, axes: a.reshape([2]*n_).transpose(axes)

            def _reorder(a, kind):
                if not correct_order:
                    match kind:
                        case 'ket':  a = _transpose(a, n, axes_new)
                        case 'dm':   a = _transpose(a, 2*n, axes_new + [i + n for i in axes_new])
                        case 'op':   a = _transpose(a, n_tot, axes_new_)
                        case 'choi': a = _transpose(a, 2*n_tot, axes_new_ + [i + n_tot for i in axes_new_])
                return a.reshape(get_shape(kind))
        else:
            _reorder = lambda a, kind: a.reshape(get_shape(kind))

        if not self.is_matrix_mode():
            self._state = _reorder(self._state, 'ket')
        else:
            self._state = _reorder(self._state, 'dm')
        if self.track_operators:
            if not self.is_superoperator:
                self._operators = [_reorder(o, 'op') for o in self._operators]
                # if not correct_order:
                #     self._operators = [_transpose(o, n_tot, axes_new_) for o in self._operators]
                # s = get_shape('op')
                # self._operators = [o.reshape(s) for o in self._operators]
            else:
                self._operators = _reorder(self._operators, 'choi')

        if not correct_order:
            # update bookkeeping
            self._qubits = tuple(new_order_all)
            self._input_qubits = [q for q in new_order_all if q in self._input_qubits] \
                               + [q for q in self._input_qubits if q not in new_order_all]

    @contextmanager
    def no_noise(self):
        """ Context manager to temporarily disable noise channel tracking. """
        prev_flag = self._noise_channel_flag
        self._noise_channel_flag = True
        try:
            yield
        finally:
            self._noise_channel_flag = prev_flag

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

    def rank(self, qubits='all', obs=None):
        qubits = self._check_qubit_arguments(qubits, False)
        if self.is_matrix_mode():
            state = self.get_state(qubits, collapse=False)
            return np.linalg.matrix_rank(state)
        if len(qubits) == self.n:
            return 1
        return self.schmidt_number(qubits, obs)  # faster than matrix_rank

    def ensemble(self, obs=None, filter_eps=None):
        """
        Returns a minimal ensemble of orthnormal kets.
        """
        if filter_eps is None:
            filter_eps = self.FILTER_EPS
        self._reorder(self._original_order, separate=False)
        with self.observable(obs):
            return ensemble_from_state(self._state, filter_eps=filter_eps, check=0)

    def ensemble_pp(self, obs=None, filter_eps=None):
        probs, kets = self.ensemble(obs, filter_eps)
        print(f"Prob      State")
        print("-"*25)
        for p, ket in zip(probs, kets):
            print(f"{p:.6f}  {unket(ket)}")

    def purify(self, obs=None):
        """
        Convert density matrix to a state vector representation using state purification and Stinespring dilation.
        """
        if not self.is_matrix_mode():
            # warn("State is already a vector")
            return self
        if self.is_superoperator:
            raise ValueError("Purification (state vector mode) is not supported for superoperator representation. Convert to Kraus mode first using `.to_kraus()`.")

        with self.observable(obs):
            # purify state
            self._reorder([], separate=False)
            state = purify(self._state, min_rank=len(self._operators), filter_eps=self.FILTER_EPS, check=0)
            d_ancilla = state.shape[0] // self._state.shape[0]
            self._state = state

            if d_ancilla == 1:  # 0 ancilla qubits
                return self

            n_ancillas = count_qubits(d_ancilla)
            if not self._too_large_to_track_operators(n_ancillas):
                # Update operators via Stinespring dilation
                V = stinespring_dilation(self._operators, min_rank=d_ancilla, check=0)
                # if self.check_level >= 2:
                #     assert is_isometry(V, kind='right'), "Stinespring dilation did not return an isometry"
                self._operators = [V]  # V is an isometry

            # add ancillas to bookkeeping
            ancillas = self._get_new_qubits_ids(n_ancillas)
            self._qubits += tuple(ancillas)
            self._original_order += ancillas
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

        if sort is None or callable(sort):
            skey = sort
            word = ""
        else:
            match sort:
                case 'in':
                    skey = lambda x: (len(x[0]), x[0])
                    word = "First "
                case 'out':
                    skey = lambda x: (len(x[1]), x[1])
                    word = "First "
                case 'asc':
                    skey = lambda x: x[2]
                    word = "Lowest "
                case 'desc':
                    skey = lambda x: -x[2]
                    word = "Highest "
                case _:
                    raise ValueError(f"Invalid sort parameter: {sort}")

        if head is None or head + 1 >= 2**len(qubits) - 1:
            howmany = "All"
        else:
            howmany = f"{word}{head}"
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
        state = self.get_state(qubits, obs=obs, allow_vector=False)  # uses von_neumann_entropy to decide if vector is possible -> avoid infinite recursion
        return von_neumann_entropy(state, check=0)

    def entanglement_entropy(self, qubits, obs=None):
        """
        Calculate the entanglement entropy of the given qubits with respect to the rest of the system.

        Alias for `von_neumann_entropy(qubits)`.
        """
        qubits = self._check_qubit_arguments(qubits, False)
        if len(qubits) == self.n:
            raise ValueError("Entanglement entropy requires a bipartition of the qubits")
        if not self.is_matrix_mode():  # take the smaller subsystem if is pure
            if len(qubits) > self.n//2:
                qubits = [q for q in self._qubits if q not in qubits]
        return self.von_neumann_entropy(qubits, obs=obs)

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

    def purity(self, qubits='all', obs=None):
        if not self.is_matrix_mode():  # take the smaller subsystem if state is pure
            qubits = self._check_qubit_arguments(qubits, False)
            if len(qubits) > self.n//2:
                qubits = [q for q in self._qubits if q not in qubits]
        return purity(self.get_state(qubits, obs=obs, allow_vector=False), check=0)

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

    def schmidt_number(self, qubits='all', obs=None):
        return len(self.schmidt_coefficients(qubits, obs))

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

    def noise(self, noise_channel='depolarizing', qubits='all', p=None, obs=None):
        """
        Apply noise to the qubits. See `noise_models.keys()` for available noise channel identifiers.
        """
        if isinstance(noise_channel, str):
            if p == 0:
                return self  # no effect
            if noise_channel not in noise_models:
                raise ValueError(f"Invalid noise model: {noise_channel}. Valid options are: " + ', '.join(noise_models.keys()))
            if noise_channel == 'zdrift':
                p = p or np.random.normal()
            else:
                assert p is None or 0 < p <= 1, f"p must be a float between 0 and 1, but was: {p}"
                p = p or self.NOISE_P
            noise_channel = noise_models[noise_channel](p)
        with self.observable(obs, qubits) as qubits:
            with self.no_noise():  # Prevent application of the noise scheduler
                return self(noise_channel, qubits)

    def apply_unitary_elementary(self, U, qubits):
        U = self.parse_unitary(U)
        n = count_qubits(U)
        qubits = self._check_qubit_arguments(qubits, False)
        assert len(qubits) == n, f"Number of qubits ({len(qubits)}) does not match unitary size ({n})"
        if n == 1: # decompose using ZYZ Euler angles
            U /= sqrt(np.linalg.det(U))
            angle_U = np.angle(U)

            gamma = 2 * np.arccos(np.clip(np.abs(U[0,0]), -1, 1))
            if np.sin(gamma/2) > 1e-12:
                beta = angle_U[1,0] - angle_U[0,0]
                delta = angle_U[0,1] - angle_U[1,0]
            else:
                beta = 0
                delta = angle_U[1,1] - angle_U[0,0]

            self.rz(beta, qubits[0])
            self.ry(gamma, qubits[0])
            self.rz(delta, qubits[0])
            return
        else:
            if "qiskit" in sys.modules:
                from qiskit import QuantumCircuit
                qcirc = QuantumCircuit(n)
                qcirc.unitary(U, list(range(n)))
                self.apply_qiskit_circuit(qcirc, qubits, elementary_only=True)
                return
            # TODO
            # 1. Matrix logarithm
            # 2. Pauli decomposition
            # 3. Trotterization (for n steps)
            raise ValueError(f"Can't decompose a unitary with more than 1 qubit, but was {n}")

    def __str__(self, sort_qubits=True):
        try:
            if sort_qubits:
                state = self.get_state()
            else:
                state = self._state.reshape(2**self.n, -1)
                if state.shape[1] == 1:
                    state = state.ravel()

            if self.is_matrix_mode():
                state = '\n' + str(state)
            else:
                n_terms = np.count_nonzero(state)
                if n_terms < 256:
                    state = f"'{unket(state)}'"
                else:
                    state = f"vector with {n_terms} terms"
        except Exception as e:
            state = f'Error: {e}'
        return f"qubits {self._qubits} in state {state}"

    def __repr__(self):
        return self.__str__(sort_qubits=False) + f" at {hex(id(self))}"

    def _repr_pretty_(self, p, cycle):  # this is used by IPython
        if cycle:
            p.text(f"{self.__class__.__name__}(...)")
        else:
            p.text(str(self))

    def _index_to_qubits(self, qubits):
        if isinstance(qubits, (slice, int)):
            return self._original_order[qubits]
        qubits = as_list_not_str(qubits)
        return [self._original_order[q] for q in qubits]

    def __getitem__(self, qubits):
        qubits = self._index_to_qubits(qubits)
        return self.get_state(qubits, allow_vector=False)

    def __setitem__(self, qubits, state):
        qubits = self._index_to_qubits(qubits)
        self.init(state, qubits)

    def __delitem__(self, qubits):
        qubits = self._index_to_qubits(qubits)
        self.remove(qubits)

    @property
    def nbytes(self):
        return nbytes(self._state) + nbytes(self._operators)

    def __neg__(self):
        if not self.is_matrix_mode():
            raise ValueError("Negation is not defined for state vectors")

        state = I_(self.n) - self._state.reshape(2**self.n, -1)
        state /= np.trace(state)
        self._state = state
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

    def tdg(self, q):
        return self(T_gate_dg, q)

    def s(self, q):
        return self(S, q)

    def sdg(self, q):
        return self(S_dg, q)

    def cx(self, control, target):
        return self(CX, [control, target])

    def cy(self, control, target):
        return self(CY, [control, target])

    def cz(self, control, target):
        return self(CZ, [control, target])

    def nx(self, control, target):
        return self(NX, [control, target])

    def ny(self, control, target):
        return self(NY, [control, target])

    def nz(self, control, target):
        return self(NZ, [control, target])

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

    def crx(self, angle, control, target):
        return self.c(Rx(angle), control, target)

    def cry(self, angle, control, target):
        return self.c(Ry(angle), control, target)

    def crz(self, angle, control, target):
        return self.c(Rz(angle), control, target)

    def p(self, angle, q):
        return self(P(angle), q)

    def cp(self, angle, control, target):
        return self.c(P(angle), control, target)

    def phase(self, angle, q):
        return self(np.exp(-1j*angle/2)*I, q)

    def su(self, phi, theta, lam, q):
        """ Always generates SU(2) unitaries. """
        return self(Rot(phi, theta, lam), q)

    def u(self, theta, phi, lam, q, include_phase=False, elementary=False):
        if elementary:
            self.rz(lam, q)
            self.ry(theta, q)
            self.rz(phi, q)
            if include_phase:
                self.phase(-(phi + lam), q)
        else:
            c = cos(theta/2)
            s = sin(theta/2)
            U = np.array([
                [c, -np.exp(1j*lam) * s],
                [np.exp(1j*phi) * s, np.exp(1j*(phi+lam)) * c]
            ], dtype=complex)
            if not include_phase:
                U *= np.exp(-1j*(phi + lam)/2)
            self(U, q)
        return self

    def qft(self, qubits, inverse=False, do_swaps=True, single_unitary=True):
        qubits = self._check_qubit_arguments(qubits, False)
        if single_unitary:
            QFT = Fourier_matrix(n=2**len(qubits), swap=not do_swaps)
            if inverse:
                QFT = QFT.T.conj()
            return self(QFT, qubits)

        # implement using elementary gates (Nielsen p. 219)
        n = len(qubits)
        if not inverse:
            Rks = [np.array([[1,0],[0,np.exp(2j*np.pi/2**k)]]) for k in range(2,n+1)]
            for i, q in enumerate(qubits):
                self.h(q)
                for j in range(n-i-1):
                    self.c(Rks[j], qubits[i+j+1], q)

        if do_swaps:
            for i in range(n//2):
                # swap
                self.cx(qubits[i], qubits[n-i-1])
                self.cx(qubits[n-i-1], qubits[i])
                self.cx(qubits[i], qubits[n-i-1])

        if inverse:
            Rks_inv = [np.array([[1,0],[0,np.exp(-2j*np.pi/2**k)]]) for k in range(2,n+1)]
            for i in reversed(range(n)):
                for j in reversed(range(n-i-1)):
                    self.c(Rks_inv[j], qubits[i+j+1], qubits[i])
                self.h(qubits[i])
        return self

    def iqft(self, qubits, do_swaps=True, single_unitary=True):
        return self.qft(qubits, True, do_swaps, single_unitary)

    def pe(self, U, state, energy, use_elementary_gate_qft=False):
        # 1. Hadamard on energy register
        self.h(energy)

        # 2. Conditioned unitary powers
        U = self.parse_unitary(U, check=self.check_level)
        for j, q in enumerate(energy):
            if j > 0:
                U = U @ U
            self.c(U, q, state)

        # 3. IQFT on energy register
        self.iqft(energy, do_swaps=False, single_unitary=not use_elementary_gate_qft)
        return self

    @staticmethod
    def parse_unitary(U, n_qubits=None, check=2):
        if isinstance(U, np.ndarray):
            pass
        elif isinstance(U, (list, tuple)):
            U = np.asarray(U)
        elif hasattr(U, 'toarray'):
            U = U.toarray()
        elif isinstance(U, str):
            U = parse_unitary(U)
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
        if isinstance(H, np.ndarray):
            pass
        elif isinstance(H, (list, tuple)):
            H = np.asarray(H)
        elif hasattr(H, 'toarray'):
            H = H.toarray()
        elif isinstance(H, str):
            H = parse_hamiltonian(H)
        else:
            raise ValueError(f"Can't process observable of type {type(H)}: {H}")
        if check >= 2:
            assert is_hermitian(H), f"Observable is not hermitian: {H}"
        if n_qubits is not None:
            n_obs = count_qubits(H)
            assert n_obs == n_qubits, f"Observable has {n_obs} qubits, but {n_qubits} qubits were provided"
        return H

    def apply_qiskit_circuit(self, circuit, qubits='all', collapse=True, elementary_only=False, include_phases=False):
        """
        Apply all gates specified in a qiskit QuantumCircuit object to QuantumComputer.
        """
        gate_map = {
            'x':      lambda self, qubits, params: self.x(qubits[0]),
            'y':      lambda self, qubits, params: self.y(qubits[0]),
            'z':      lambda self, qubits, params: self.z(qubits[0]),
            'h':      lambda self, qubits, params: self.h(qubits[0]),
            's':      lambda self, qubits, params: self.s(qubits[0]),
            't':      lambda self, qubits, params: self.t(qubits[0]),
            'sdg':    lambda self, qubits, params: self.sdg(qubits[0]),
            'tdg':    lambda self, qubits, params: self.t_dg(qubits[0]),
            'cx':     lambda self, qubits, params: self.cx(qubits[0], qubits[1]),
            'rx':     lambda self, qubits, params: self.rx(params[0], qubits[0]),
            'ry':     lambda self, qubits, params: self.ry(params[0], qubits[0]),
            'rz':     lambda self, qubits, params: self.rz(params[0], qubits[0]),
            'p':      lambda self, qubits, params: self.p(params[0], qubits[0]),
            'u':      lambda self, qubits, params: self.u(*params, qubits[0],
                                include_phase=include_phases, elementary=elementary_only),
        }

        if not elementary_only:
            gate_map.update({
                'cy':     lambda self, qubits, params: self.cy(qubits[0], qubits[1]),
                'cz':     lambda self, qubits, params: self.cz(qubits[0], qubits[1]),
                'cp':     lambda self, qubits, params: self.cp(params[0], qubits[0], qubits[1]),
                'swap':   lambda self, qubits, params: self.swap(qubits[0], qubits[1]),
                'cswap':  lambda self, qubits, params: self.cswap(qubits[0], qubits[1], qubits[2]),
                'ccx':    lambda self, qubits, params: self.ccx(qubits[0], qubits[1], qubits[2]),
            })

        def _from_qiskit_inner(circuit, qubit_map, measurement_results):
            for el in circuit.data:
                instr, qargs, cargs = el.operation, el.qubits, el.clbits
                name = instr.name.lower()
                qubits = [qubit_map[q] for q in qargs]
                if name in gate_map:
                    # print("Apply", name, qubits, getattr(instr, 'params', []))
                    gate_map[name](self, qubits, getattr(instr, 'params', []))
                elif name == 'measure':
                    result = self.measure(qubits, collapse=collapse, return_as='binstr')
                    if collapse:
                        for idx, cbit in enumerate(cargs):
                            measurement_results[cbit._index] = result[idx]
                elif name == 'unitary':
                    self(instr.to_matrix(), qubits)
                elif name == 'qft':
                    if not instr.definition:
                        raise ValueError("QFT gate has no valid definition")
                    do_swaps = instr.definition.data[-1].operation.name.lower() == 'swap'
                    self.qft(qubits[::-1], do_swaps=do_swaps, single_unitary=not elementary_only)
                elif instr.definition:
                    # print("Unknown:", name)
                    _from_qiskit_inner(
                        instr.definition,
                        {subq: qubits[i] for i, subq in enumerate(instr.definition.qubits)},
                        measurement_results
                    )
                elif name.startswith("save_"):
                    # Ignore Qiskit instructions that start with "save_", e.g. "save_unitary"
                    continue
                else:
                    raise NotImplementedError(f"Gate '{name}' not supported and cannot be decomposed further.")
            return

        qubit_list = list(circuit.qubits)
        n = len(qubit_list)
        # Allocate/check qubits if needed
        qubits, to_alloc = self._check_qubit_arguments(qubits, True)
        if self.n == 0:
            qubits = to_alloc = list(range(n))
        self._alloc_qubits(to_alloc)
        assert len(qubits) == n, f"Provided qubits length {len(qubits)} does not match the number of qubits in the given circuit {n}"
        qubit_map = {q: qubits[i] for i, q in enumerate(qubit_list)}
        n_clbits = len(circuit.clbits)
        measurement_results = ['0'] * n_clbits
        _from_qiskit_inner(circuit, qubit_map, measurement_results)
        if n_clbits != 0 and collapse:
            return ''.join(measurement_results)
        return None

def create_benchmark_noise_scheduler(
    P1=3e-4,
    P2=5e-3,
    PAD=2e-4,
    DRIFT_SIGMA=0.01,
    P_MEAS=2e-2,
    P_IDLE=5e-5,
    depolarizing_scaling='linear'
):
    """
    Factory for a noise scheduler for the QuantumComputer class.

    Parameters
    ----------
    P1 : float
        1-qubit depolarising probability.
    P2 : float
        2-qubit depolarising probability (also used as base for multi-qubit gates).
    PAD : float
        Amplitude-damping probability per gate.
    DRIFT_SIGMA : float
        Standard deviation for Z-phase drift (radians), re-sampled once per shot.
    P_MEAS : float
        Measurement bit-flip probability.
    P_IDLE : float
        Idle depolarizing + amplitude-damping probability per gate.
    depolarizing_scaling : str
        'none'      for Pk = P2  (constant)
        'linear'    for Pk = P2 * (k-1)
        'quadratic' for Pk = P2 * (k-1) * k / 2

    Returns
    -------
    benchmark_noise_scheduler : function
        A noise scheduler with the specified parameters.
    """
    assert 0 <= P1 <= 1, f"P1 must be between 0 and 1, got {P1}"
    assert 0 <= P2 <= 1, f"P2 must be between 0 and 1, got {P2}"
    assert 0 <= PAD <= 1, f"PAD must be between 0 and 1, got {PAD}"
    assert DRIFT_SIGMA >= 0, f"DRIFT_SIGMA must be non-negative, got {DRIFT_SIGMA}"
    assert 0 <= P_MEAS <= 1, f"P_MEAS must be between 0 and 1, got {P_MEAS}"
    assert 0 <= P_IDLE <= 1, f"P_IDLE must be between 0 and 1, got {P_IDLE}"
    assert depolarizing_scaling in ('linear', 'quadratic', 'none'), f"depolarizing_scaling must be 'linear', 'quadratic', or 'none', but got {depolarizing_scaling}"

    def benchmark_noise_scheduler(qubits, process_type, qc: QuantumComputer):
        if process_type == 'init' and DRIFT_SIGMA:
            qc._drift_angle = np.random.normal(0, DRIFT_SIGMA)
        elif process_type == 'apply':
            k = len(qubits)
            Pk = P1
            if k > 1 and P2:
                Pk = P2
                if depolarizing_scaling == 'linear':
                    Pk = P2*(k-1)
                elif depolarizing_scaling == 'quadratic':
                    Pk = P2*(k-1)*k/2
                Pk = min(Pk, 1.0)
            qc.noise('depolarizing', qubits, p=Pk)
            qc.noise('amplitude_damping', qubits, p=PAD)
            if hasattr(qc, '_drift_angle'):
                qc.noise('zdrift', qubits, p=qc._drift_angle)
            if P_IDLE:
                others = [q for q in qc.qubits if q not in qubits]
                if others:
                    qc.noise('depolarizing', others, p=P_IDLE)
                    qc.noise('amplitude_damping', others, p=P_IDLE)
        elif process_type == 'measure' and P_MEAS:
            return noise_models['bitflip'](p=P_MEAS)
    return benchmark_noise_scheduler
