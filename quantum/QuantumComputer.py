import psutil, warnings
from contextlib import contextmanager
import numpy as np
import scipy.sparse as sp

from .constants import *
from .state import partial_trace, ket, op, dm, unket, count_qubits, random_ket, plotQ
from .hamiltonian import parse_hamiltonian
from .unitary import parse_unitary, get_unitary, Fourier_matrix
from .info import entropy_entanglement
from ..mathlib import choice, normalize, binstr_from_int, bipartitions, is_hermitian
from ..plot import imshow
from ..utils import is_int, duh

class QuantumComputer:
    """
    A naive simulation of a quantum computer. Can simulate as state vector or density matrix.
    """
    def __init__(self, qubits=None, state=None, track_unitary='auto'):
        self.track_unitary = track_unitary
        self.MATRIX_SLOW = 8
        self.MATRIX_BREAK = 12
        self.clear()

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
        if self._track_unitary:
            qc.U = self.U.copy()
        return qc

    def __call__(self, U, qubits='all'):
        qubits, to_alloc = self._check_qubit_arguments(qubits, True)
        self._alloc_qubits(to_alloc)
        U = self.parse_unitary(U)
        if U.shape == (2,2) and len(qubits) > 1:
            U_ = U
            for _ in qubits[1:]:
                U = np.kron(U, U_)
        assert U.shape == (2**len(qubits), 2**len(qubits)), f"Invalid unitary shape for {len(qubits)} qubits: {U.shape} != {2**len(qubits),2**len(qubits)}"
        # rotate axes of state vector to have the `qubits` first
        self._reorder(qubits)

        # apply unitary
        if self.is_matrix_mode():
            if len(qubits) == self.n:
                # (q x q) x (q x q) x (q x q) -> q x q
                self.state = U @ self.state @ U.T.conj()
            else:
                # (q x q) x (q x (n-q) x q x (n-q)) x (q x q) -> q x (n-q) x q x (n-q)
                self.state = np.tensordot(U, self.state, axes=1)
                # (q x (n-q) x q x (n-q)) x (q x q) -> q x (n-q) x (n-q) x q
                self.state = np.tensordot(self.state, U.T.conj(), axes=(2,0))
                # q x (n-q) x (n-q) x q -> q x (n-q) x q x (n-q)
                self.state = self.state.transpose([0, 1, 3, 2])
        else:
            # (q x q) x (q x (n-q)) -> q x (n-q)  or  (q x q) x q -> q
            self.state = np.tensordot(U, self.state, axes=1)

        # update unitary if tracked
        if self._track_unitary:
            # (q x q) x (q x q x 2(n-q)) -> q x q x 2(n-q)
            self.U = np.tensordot(U, self.U, axes=1)
        return self

    def get_state(self, qubits='all'):
        return self._get("state", qubits)

    def get_U(self, qubits='all'):
        if not self._track_unitary:
            raise ValueError("Unitary tracking is disabled")
        return self._get("U", qubits)

    def _get(self, prop, qubits):
        if self.n == 0:
            raise ValueError("No qubits allocated yet")
        qubits = self._check_qubit_arguments(qubits, False)
        self._reorder(qubits, reshape=False)
        a = getattr(self, prop)
        if len(qubits) == self.n:
            return a
        return partial_trace(a, [self.qubits.index(q) for q in qubits])

    @contextmanager
    def observable(self, obs=None, qubits='all'):
        qubits = self._check_qubit_arguments(qubits, False)
        if obs is not None:
            obs = self.parse_hermitian(obs, len(qubits))
            D, U = np.linalg.eigh(obs)  # eigh produces less numerical errors than eig
            self(U.T.conj(), qubits)  # basis change
        try:
            yield qubits
        finally:
            if obs is not None:
                self(U, qubits)  # back to standard basis

    def probs(self, qubits='all', obs=None):
        with self.observable(obs, qubits) as qubits:
            return self._probs(qubits)

    def _probs(self, qubits='all'):
        if self.is_matrix_mode():
            state = self.get_state(qubits)
            return np.diag(state).real
        else:
            self._reorder(qubits)
            probs = np.abs(self.state)**2
            if len(probs.shape) == 1:  # all qubits
                return probs
            return np.sum(probs, axis=1)

    def measure(self, qubits='all', collapse=True, obs=None):
        self._reset_unitary()
        with self.observable(obs, qubits) as qubits:
            probs = self._probs(qubits)
            q = len(qubits)
            if self.is_matrix_mode():
                self.state = self.state.reshape(2**self.n, 2**self.n)
                if collapse:
                    outcome = choice(range(2**q), p=probs)

                    # P = np.kron(op(outcome, n=q), I_(self.n-q))  # projector
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
                    #     Pi = np.kron(op(i, n=q), I_(self.n-q))
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
                if not collapse:
                    # self._reorder(qubits)  # already done in _probs
                    # if the state is a pure state, we can keep it as is
                    # if entropy(probs) < 1e-12:
                    #     print("Warning: Deterministic outcome -> not entangled -> no decoherence required")
                    #     return self
                    if self.n > self.MATRIX_BREAK:
                        warnings.warn("collapse=False for large n -> using vector collapse (collapse=True) instead of density matrix")
                        collapse = True
                    else:
                        # repeat as density matrix
                        self.state = np.outer(self.state, self.state.conj())
                        return self.measure(qubits, collapse=collapse, obs=obs)
                if collapse:
                    # play God
                    outcome = np.random.choice(2**q, p=probs)
                    # collapse
                    keep = self.state[outcome]
                    self.state = np.zeros_like(self.state)
                    self.state[outcome] = normalize(keep)  # may be 1 or vector

        if not collapse:
            return self
        return binstr_from_int(outcome, len(qubits))

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
        if not self.is_matrix_mode():
            q = len(qubits)
            new_state = ket(0, n=q)
            if q == self.n:
                self.state = new_state
                self.qubits = qubits
                return self
            probs = self._probs(qubits)  # also moves qubits to the front and reshapes
            outcome = np.random.choice(2**q, p=probs)
            keep = self.state[outcome] / np.sqrt(probs[outcome])
            self.state = np.zeros_like(self.state)
            self.state[outcome] = keep
            return self
        else:
            raise ValueError("Special reset not available for matrix mode")

    def init(self, state, qubits=None, collapse=True):
        if qubits is None:
            if self.n == 0:  # infer `qubits` from `state`
                if hasattr(state, 'shape') and len(state.shape) == 2:
                    self.state = dm(state)
                else:
                    self.state = ket(state)
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
            new_state = dm(state, n=len(qubits))
            if self.n > 0 and not self.is_matrix_mode():
                # switch to matrix mode
                self.state = op(self.state)
        else:
            if isinstance(state, str) and state == 'random_pure':
                state = 'random'
            if self.is_matrix_mode():
                new_state = dm(state, n=len(qubits))
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
        self.init(random_ket(n))
        return self

    def add(self, qubits, state=None):
        if state is None:
            qubits, to_alloc = self._check_qubit_arguments(qubits, True)
            self._alloc_qubits(to_alloc)
        else:
            self.init(state, qubits)
        return self

    def remove(self, qubits, collapse=False):
        qubits = self._check_qubit_arguments(qubits, False)
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
                self.state = partial_trace(self.state, retain)
        else:
            if collapse or entropy_entanglement(self.state.reshape(-1), qubits_indcs) < 1e-10:
                # if no entanglement with others, just remove it
                probs = self._probs(qubits)  # also moves qubits to the front and reshapes
                outcome = choice(2**len(qubits), p=probs)
                self.state = normalize(self.state[outcome])
                if len(qubits) == self.n - 1:
                    self.state = self.state.reshape([2])
            else:
                # otherwise, we need to decohere
                if len(retain) > self.MATRIX_BREAK:
                    warnings.warn("Decoherence from state vector for large n -> using vector collapse (collapse=True) instead of decoherence")
                    return self.remove(qubits, collapse=True)
                self.state = partial_trace(self.state, retain)

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

    def plot(self, show_qubits='all', **kw_args):
        state = self.get_state(show_qubits)
        if len(state.shape) == 2:
            return imshow(state, **kw_args)
        return plotQ(state, **kw_args)

    def plotU(self, show_qubits='all', **kw_args):
        U = self.get_U(show_qubits)
        return imshow(U, **kw_args)

    def _check_qubit_arguments(self, qubits, allow_new):
        if isinstance(qubits, slice):
            qubits = self.qubits[qubits]
        elif isinstance(qubits, str) and qubits == 'all':
            qubits = self.original_order
        elif not isinstance(qubits, (list, tuple, np.ndarray)):
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
                warnings.warn(f"Insufficient RAM ({self.n + q}-qubit unitary would require {duh(RAM_required)})")
        else:
            if self.is_matrix_mode():  # False if self.n == 0
                RAM_required = (2**(self.n + q))**2*16*2
            else:
                RAM_required = (2**(self.n + q))*16*2
            if RAM_required > psutil.virtual_memory().available:
                warnings.warn(f"Insufficient RAM ({self.n + q}-qubit state would require {duh(RAM_required)})")

        if self.is_matrix_mode():
            state = state if state is not None else op(0, n=q)
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

        def _reorder(a, axes_new, is_matrix):
            # alternatively, is_matrix = np.prod(a.shape) == 2**(2*self.n)
            if is_matrix:
                axes_new = axes_new + [i + self.n for i in axes_new]
            m = 2 if is_matrix else 1
            if any(s > 2 for s in a.shape):
                a = a.reshape([2]*m*self.n)
            a = a.transpose(axes_new)
            # collect
            if reshape and n_front < self.n:
                a = a.reshape([2**n_front, 2**(self.n-n_front)]*m)
            else:
                a = a.reshape([2**self.n]*m)
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

    def ensemble(self):
        self._reorder(self.original_order)

        if self.is_matrix_mode():
            probs, kets = np.linalg.eigh(self.state)
            # filter out zero eigenvalues
            mask = probs > 1e-12
            probs = probs[mask]
            kets = kets[:, mask]
            return probs.real, kets
        else:
            return np.array([1.]), np.array([self.state]).T

    def purify(self, sample=False):
        """
        Convert density matrix to a state vector representation by purification, either by doubling the number of qubits or by sampling from the eigenstates.
        """
        if not self.is_matrix_mode():
            warnings.warn("State is already a vector")
            return self

        probs, kets = self.ensemble()
        if sample or len(probs) == 1:
            outcome = choice(len(probs), p=normalize(probs, p=1))
            new_state = kets[:, outcome]
            n_ancillas = 0
        else:
            # construct purification
            n_ancillas = int(np.ceil(np.log2(len(probs))))
            new_state = np.zeros(2**(self.n + n_ancillas), dtype=complex)
            ancilla_basis = I_(n_ancillas)

            for i, p in enumerate(probs):
                new_state += np.sqrt(p) * np.kron(kets[:, i], ancilla_basis[i])

        # find n_ancillas integers that are not in self.qubits
        ancillas = []
        i = self.n
        while len(ancillas) < n_ancillas:
            while i in self.qubits:
                i += 1
            ancillas.append(i)

        # initialize the new purified state
        self.init(new_state, self.qubits + ancillas)
        return self

    def to_dm(self):
        """
        Convert state vector to density matrix representation, if sufficient RAM is available.
        """
        if self.is_matrix_mode():
            warnings.warn("State is already a density matrix")
            return self
        # RAM check
        RAM_required = 2**(2*self.n)*16*2
        if RAM_required > psutil.virtual_memory().available:
            warnings.warn(f"Insufficient RAM ({2*self.n}-qubit density matrix would require {duh(RAM_required)})")
        else:
            self.state = np.outer(self.state, self.state.conj())
        return self

    def ev(self, obs, qubits='all'):
        qubits = self._check_qubit_arguments(qubits, False)
        state = self.get_state(qubits)
        obs = self.parse_hermitian(obs, len(qubits))
        if len(state.shape) == 2:
            ev = np.trace(state @ obs)
        else:
            ev = state.conj().T @ obs @ state
        return ev.real

    def std(self, obs, qubits='all', return_ev=False):
        qubits = self._check_qubit_arguments(qubits, False)
        state = self.get_state(qubits)
        obs = self.parse_hermitian(obs, len(qubits))
        if len(state.shape) == 2:
            m1 = np.trace(state @ obs)
            m2 = np.trace(state @ obs @ obs)
        else:
            m1 = state.conj().T @ obs @ state
            m2 = state.conj().T @ obs @ obs @ state
        var = m2 - m1**2
        std = np.sqrt(var.real)
        if return_ev:
            return std, m1
        return std

    def entanglement_entropy(self, qubits='all'):
        qubits = self._check_qubit_arguments(qubits, False)
        if len(qubits) < self.n:
            return entropy_entanglement(self.state, [self.qubits.index(q) for q in qubits])
        else:
            return [(i, o, entropy_entanglement(self.state, i))
                for i, o in bipartitions(range(self.n), unique=self.is_pure())  # H(A) = H(B) if AB is pure
            ]

    def entanglement_entropy_pp(self, sort='in', head=100, precision=7):
        res = self.entanglement_entropy()
        if sort == 'in':
            skey = lambda x: (len(x[0]), x[0])
        elif sort == 'out':
            skey = lambda x: (len(x[1]), x[1])
        elif sort == 'asc':
            skey = lambda x: x[2]
        elif sort == 'desc':
            skey = lambda x: -x[2]
        elif type(sort) == 'function':
            skey = sort
        else:
            raise ValueError(f"Invalid sort parameter: {sort}")

        howmany = "All" if head is None or head + 1 >= 1 << self.n - 1 else f"Top {head}"
        print(howmany + " bipartitions:\n" + "-"*(self.n*2+3))
        for part_in, part_out, entanglement in sorted(res, key=skey)[:head]:
            part_in  = [str(self.qubits[i]) for i in part_in]
            part_out = [str(self.qubits[i]) for i in part_out]
            print(f"{' '.join(part_in)}  |  {' '.join(part_out)} \t{entanglement:.{precision}f}".rstrip('0'))

    def schmidt_decomposition(self, qubits='all'):
        """
        Schmidt decomposition of a bipartition of the qubits. Returns the Schmidt coefficients and the two sets of basis vectors.

        >>> qc = QuantumComputer(2, '00 + 01')
        >>> U, S, V = qc.schmidt_decomposition([1])
        >>> print(S)  # [1.] because it is a product state
        [1.]
        >>> np.allclose(U @ np.diag(S) @ V.T.conj(), qc.get_state([1]))
        """
        qubits = self._check_qubit_arguments(qubits, False)
        if self.is_matrix_mode():
            raise ValueError("Schmidt decomposition is not available for density matrices")
        if len(qubits) == self.n or len(qubits) == 0:
            raise ValueError("Schmidt decomposition requires a bipartition of the qubits")

        state = self.get_state(qubits)  # state has now shape (2**q, 2**(n-q))
        U, S, V = np.linalg.svd(state, full_matrices=False)  # U = q basis, S = Schmidt coefficients, V = (n-q basis).T.conj()
        # remove zero coefficients
        Schmidt_coefficients = S[S > 1e-12]
        U = U[:, :len(Schmidt_coefficients)]
        V = V[:len(Schmidt_coefficients), :]
        return U, Schmidt_coefficients, V.T.conj()

    def schmidt_coefficients(self, qubits='all'):
        return self.schmidt_decomposition(qubits)[1]

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
        return self.__str__()

    def __getitem__(self, qubits):
        state = self.get_state(qubits)
        if len(state.shape) == 1:
            state = np.outer(state, state.conj())
        return state

    def __setitem__(self, qubits, state):
        self.init(state, qubits)
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
        U = self.parse_unitary(U, len(target))
        for _ in control:
            U = C_(U, negative=negative)
        return self(U, control + target)

    def cc(self, U, control1, control2, target):
        return self.c(U, [control1, control2], target)

    def swap(self, qubit1, qubit2):
        return self(SWAP, [qubit1, qubit2])

    def rx(self, angle, q):
        return self(Rx(angle), q)

    def ry(self, angle, q):
        return self(Ry(angle), q)

    def rz(self, angle, q):
        return self(Rz(angle), q)

    def qft(self, qubits, inverse=False, do_swaps=True):
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
        U = self.parse_unitary(U)
        UD, UU = np.linalg.eig(U)
        for j, q in enumerate(energy):
            U_2j = UU @ np.diag(UD**(2**j)) @ UU.T.conj()
            self.c(U_2j, q, state)

        # 3. IQFT on energy register
        self.iqft(energy, do_swaps=False)
        return self

    @staticmethod
    def parse_unitary(U, n_qubits=None):
        if isinstance(U, (list, np.ndarray)):
            U = np.array(U, copy=False)
        elif isinstance(U, str):
            U = parse_unitary(U)
        elif sp.issparse(U):
            U = U.toarray()
        else:
            try:
                # qiskit might not be loaded
                U = get_unitary(U)
            except:
                raise ValueError(f"Can't process unitary of type {type(U)}: {U}")
        # assert is_unitary(U), f"Unitary is not unitary: {U}"
        if n_qubits is not None:
            n_obs = count_qubits(U)
            assert n_obs == n_qubits, f"Unitary has {n_obs} qubits, but {n_qubits} qubits were provided"
        return U

    @staticmethod
    def parse_hermitian(H, n_qubits=None):
        if isinstance(H, (list, np.ndarray)):
            H = np.array(H)
        elif isinstance(H, str):
            H = parse_hamiltonian(H)
        elif sp.issparse(H):
            H = H.toarray()
        else:
            raise ValueError(f"Can't process observable of type {type(H)}: {H}")
        assert is_hermitian(H), f"Observable is not hermitian: {H}"
        if n_qubits is not None:
            n_obs = count_qubits(H)
            assert n_obs == n_qubits, f"Observable has {n_obs} qubits, but {n_qubits} qubits were provided"
        return H

def evolve(state, U, checks=True):
    if checks:
        if not hasattr(state, 'shape'):
            state = np.array(state)
        n = count_qubits(state)
        U = QuantumComputer.parse_unitary(U, n)
    if len(state.shape) == 1:
        if checks:
            state = ket(state)
        return U @ state
    if checks:
        state = dm(state)
    return U @ state @ U.T.conj()