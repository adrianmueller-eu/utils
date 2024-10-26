import psutil, warnings
import numpy as np
import itertools
from functools import reduce
import matplotlib.pyplot as plt
import scipy.sparse as sp
from .mathlib import *
from .plot import colorize_complex
from .utils import duh, is_int
from .prob import entropy

#################
### Unitaries ###
#################

fs = lambda x: 1/np.sqrt(x)
f2 = fs(2)
I_ = lambda n: np.eye(2**n)
I = I_(1)
X = np.array([ # 1j*Rx(pi)
    [0, 1],
    [1, 0]
], dtype=complex)
Y = np.array([ # 1j*Ry(pi)
    [0, -1j],
    [1j,  0]
], dtype=complex)
Z = np.array([ # 1j*Rz(pi)
    [1,  0],
    [0, -1]
], dtype=complex)
S = np.array([ # np.sqrt(Z)
    [1,  0],
    [0, 1j]
], dtype=complex)
T_gate = np.array([ # avoid overriding T = True
    [1,  0],
    [0,  np.sqrt(1j)]
], dtype=complex)
H = H_gate = 1/np.sqrt(2) * np.array([ # Fourier_matrix(2) = f2*(X + Z) = 1j*f2*(Rx(pi) + Rz(pi))
    [1,  1],
    [1, -1]
], dtype=complex)

def R_(gate, theta):
   return matexp(-1j*gate*theta/2)

Rx = lambda theta: R_(X, theta)
Ry = lambda theta: R_(Y, theta)
Rz = lambda theta: R_(Z, theta)
for i in [2,3]:
    for s, g in zip(itertools.product(['I','X','Y','Z'], repeat=i), itertools.product([I,X,Y,Z], repeat=i)):
        globals()["".join(s)] = reduce(np.kron, g)  # II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ

def C_(A, reverse=False, negative=False):
    if not hasattr(A, 'shape'):
        A = np.array(A, dtype=complex)
    n = int(np.log2(A.shape[0]))
    op0, op1 = [[1,0],[0,0]], [[0,0],[0,1]]
    if negative:
        op0, op1 = op1, op0
    if reverse:
        return np.kron(I_(n), op0) + np.kron(A, op1)
    return np.kron(op0, I_(n)) + np.kron(op1, A)
CNOT = CX = C_(X) # 0.5*(II + ZI - ZX + IX)
XC = C_(X, reverse=True)
CZ = ZC = C_(Z)
CY = C_(Y)
NX = C_(X, negative=True)
XN = C_(X, negative=True, reverse=True)
NZ = C_(Z, negative=True)
ZN = C_(Z, negative=True, reverse=True)
Toffoli = C_(C_(X))
SWAP = np.array([ # 0.5*(XX + YY + ZZ + II), CX @ XC @ CX
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)
iSWAP = np.array([ # 0.5*(1j*(XX + YY) + ZZ + II), R_(XX+YY, -pi/2)
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)
fSWAP = SWAP @ CZ

def Fourier_matrix(n, n_is_qubits=True):
    """Calculate the Fourier matrix of size `n`. The Fourier matrix is the matrix representation of the quantum Fourier transform (QFT)."""
    if n_is_qubits:
        n = 2**n
    return 1/np.sqrt(n) * np.array([[np.exp(2j*np.pi*i*j/n) for j in range(n)] for i in range(n)], dtype=complex)

def parse_unitary(unitary):
    """Parse a string representation of a unitary into its matrix representation. The result is guaranteed to be unitary.
    A universal set of quantum gates is given with 'H', 'T', 't', and 'CX'.

    Example:
    >>> parse_unitary('CX @ XC @ CX') # SWAP
    array([[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
           [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
           [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]
           [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]])
    >>> parse_unitary('SS @ HI @ CX @ XC @ IH') # iSWAP
    array([[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
           [ 0.+0.j  0.+0.j  0.+1.j  0.+0.j]
           [ 0.+0.j  0.+1.j  0.+0.j  0.+0.j]
           [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]])
    """
    def s(chunk):
        gates = []  # set dtype
        for c in chunk:
            # positive and negative controls
            if c == "C":
                gate = op(1)
            elif c == "N":
                gate = op(0)
            # single-qubit gates
            elif c == "T":
                    gate = T_gate
            elif c == "t":
                    gate = T_gate.conj()
            elif c == "s":
                    gate = S.conj()
            else:
                    gate = globals()[c]
            gates.append(gate)

        chunk_matrix = reduce(np.kron, gates)
        if "C" in chunk or "N" in chunk:
            n = len(chunk)
            part_matrix = np.array([[1]])
            no_control = 0
            for i, c in enumerate(chunk):
                if c == "C" or c == "N":
                    part_matrix = np.kron(part_matrix, I_(no_control))
                    no_control = 0
                    on, off = (op(1), op(0)) if c == "C" else (op(0), op(1))
                    chunk_matrix += np.kron(part_matrix, np.kron(off, I_(n-i-1)))
                    part_matrix = np.kron(part_matrix, on)
                else:
                    no_control += 1

        return chunk_matrix

    # Remove whitespace
    unitary = unitary.replace(" ", "")

    # Parse the unitary
    chunks = unitary.split("@")
    # Remove empty chunks
    chunks = [c for c in chunks if c != ""]
    # Use the first chunk to determine the number of qubits
    n = len(chunks[0])

    U = I_(n)
    for chunk in chunks:
        # print(chunk, unitary)
        chunk_matrix = None
        if chunk == "":
            continue
        if len(chunk) != n:
            raise ValueError(f"Gate count must be {n} but was {len(chunk)} for chunk \"{chunk}\"")

        # Get the matrix representation of the chunk
        chunk_matrix = s(chunk)

        # Update the unitary
        # print("chunk", chunk, unitary, chunk_matrix)
        U = U @ chunk_matrix

    assert is_unitary(U), f"Result is not unitary: {U, U @ U.conj().T}"

    return U

try:
    ##############
    ### Qiskit ###
    ##############

    from qiskit import transpile
    from qiskit_aer import Aer
    from qiskit import QuantumCircuit

    # Other useful imports
    from qiskit.visualization import plot_histogram

    def run(circuit, shots=1, generate_state=True, plot=True, showqubits=None, showcoeff=True, showprobs=True, showrho=False, figsize=(16,4), title=""):
        if shots > 10:
            n_gates = count_gates(circuit)
            print("#gates: %d, expected running time: %.3fs" % (n_gates, n_gates * 0.01))
        if generate_state:
            simulator = Aer.get_backend("statevector_simulator")
            if shots is None or shots == 0:
                warnings.warn("shots=0 is not supported for statevector_simulator. Using shots=1 instead.")
                shots = 1
        else:
            simulator = Aer.get_backend('aer_simulator')
        t_circuit = transpile(circuit, simulator)
        result = simulator.run(t_circuit, shots=shots).result()

        if generate_state:
            state = np.array(result.get_statevector())
            # qiskit outputs the qubits in the reverse order
            state = reverse_qubit_order(state)
            if plot:
                plotQ(state, showqubits=showqubits, showcoeff=showcoeff, showprobs=showprobs, showrho=showrho, figsize=figsize, title=title)
            return result, state
        else:
            return result

    class exp_i(QuantumCircuit):
        def __init__(self, H, k=1, use_pauli=False, trotter_steps=3):
            """
            Represents a quantum circuit for $e^{iH}$ for a given hamiltonian (hermitian matrix) $H \\in \\mathbb C^{d\\times d}$.
            If `use_pauli == False`, it calculates full diagonalization and stores two dxd matrices (+ one dx1 vector).
            If `use_pauli == True`, `PauliEvolutionGate` is used, which has lower memory footprint if the Hamiltonian can be expressed
            in a polynomial number of Pauli terms. If the system is large this also generates shorter (but only approximative) circuits.
            """
            from qiskit.synthesis.evolution import SuzukiTrotter
            from qiskit.quantum_info import SparsePauliOp
            from qiskit.circuit.library import PauliEvolutionGate
            from qiskit.quantum_info.operators import Operator

            self.use_pauli = use_pauli
            if self.use_pauli:
                k = -k  # PauliEvolutionGate uses exp(-iHt)
                synth = SuzukiTrotter(2*trotter_steps)
                if isinstance(H, PauliEvolutionGate):
                    self.pauli_ev = H
                elif isinstance(H, SparsePauliOp):
                    self.pauli_ev = PauliEvolutionGate(H, k, synthesis=synth)
                elif isinstance(H, dict):
                    pauli = SparsePauliOp.from_list(H.items())
                    self.pauli_ev = PauliEvolutionGate(pauli, k, synthesis=synth)
                else:
                    coeffs, obs = pauli_decompose(H)
                    pauli = SparsePauliOp.from_list(zip(obs, coeffs))
                    self.pauli_ev = PauliEvolutionGate(pauli, k, synthesis=synth)
                self.n = self.pauli_ev.num_qubits
            else:
                if isinstance(H, tuple):
                    self.D, self.U = H
                    self.n = count_qubits(self.D)
                else:
                    self.n = count_qubits(H)
                    # diagonalize, if necessary
                    if self.n >= 12:
                        warnings.warn(f"Diagonalizing a {self.n}-qubit matrix")
                    self.D, self.U = np.linalg.eigh(H)
            # auxiliary variables
            try:
                self.k = int(k)
            except:
                self.k = k
            name = "exp^-i" if k < 0 else "exp^i"
            name += "H" if abs(k) == 1 else f"{abs(k)}H"
            super().__init__(self.n, name=name)       # circuit on n qubits
            self.all_qubits = list(range(self.n))
            # calculate and add unitary
            if use_pauli:
                self.append(self.pauli_ev, self.all_qubits)
            else:
                u = self.get_unitary()
                self.unitary(Operator(u), self.all_qubits, label=self.name)

        def power(self, k):
            if self.use_pauli:
                if k == -1:
                    return self.pauli_ev.inverse()
                return exp_i(self.pauli_ev, k=-self.k*k)
            return exp_i((self.D, self.U), k=self.k*k, use_pauli=False)  # the tuple only stores the references

        def __pow__(self, k):
            return self.power(k)

        def inverse(self):
            if self.use_pauli:
                return self.pauli_ev.inverse()
            return exp_i((self.D, self.U), k=-self.k, use_pauli=False)

        def get_unitary(self, k=1):
            if self.use_pauli:
                qc = QuantumCircuit(self.n)
                qc.append(self, self.all_qubits)
                return get_unitary(qc)
            return self.U @ np.diag(np.exp(self.k*k*1j*self.D)) @ self.U.T.conj()

    def get_unitary(circ, decimals=None, as_np=True):
        if hasattr(circ, 'get_unitary'):
            return circ.get_unitary()
        sim = Aer.get_backend('unitary_simulator')
        t_circuit = transpile(circ, sim)
        res = sim.run(t_circuit).result()
        U   = res.get_unitary(decimals=decimals)
        if as_np:
            return np.array(U)
        return U

    def get_pe_energies(U):
        if isinstance(U, QuantumCircuit):
            U = get_unitary(U)
        eigvals, eigvecs = np.linalg.eig(U)
        energies = np.angle(eigvals)/(2*np.pi)
        return energies

    def show_eigenvecs(U, showrho=False):
        if isinstance(U, QuantumCircuit):
            U = get_unitary(U)
        eigvals, eigvecs = np.linalg.eig(U)
        print(np.round(eigvecs, 3))
        for i in range(eigvecs.shape[1]):
            plotQ(eigvecs[:,i], figsize=(12,2), showrho=showrho)
        return eigvecs

    def count_gates(qc, decompose_iterations=4, isTranspiled=False):
        if not isTranspiled:
            simulator = Aer.get_backend('aer_simulator')
            t_circuit = transpile(qc, simulator)
        else:
            t_circuit = qc
        for _ in range(decompose_iterations):
            t_circuit = t_circuit.decompose()
        return len(t_circuit._data)

except ModuleNotFoundError:
    print("Warning: qiskit not installed! Use `pip install qiskit qiskit_aer pylatexenc`.")
    pass

class QuantumComputer:
    """
    A naive simulation of a quantum computer.
    """
    def __init__(self, qubits=None, state=None, track_unitary='auto'):
        self.track_unitary = track_unitary
        self.MAX_UNITARY = 8
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
            self.track_unitary == 'auto' and self.n > self.MAX_UNITARY
        )

    def clear(self):
        self.state = np.array([1.])
        self.qubits = []
        self.original_order = []
        self._reset_unitary()
        return self

    def __call__(self, U, qubits):
        qubits = self._check_qubit_arguments(qubits, True)
        U = self.parse_unitary(U)
        if U.shape == (2,2) and len(qubits) > 1:
            U_ = U
            for _ in qubits[1:]:
                U = np.kron(U, U_)
        assert U.shape == (2**len(qubits), 2**len(qubits)), f"Invalid unitary shape for {len(qubits)} qubits: {U.shape} != {2**len(qubits),2**len(qubits)}"
        # rotate axes of state vector to have the `qubits` first
        self._reorder(qubits)
        # apply U: (q x q) x (q x (n-q)) -> q x (n-q)
        self.state = U @ self.state
        if self._track_unitary:
            self.U = np.tensordot(U, self.U, axes=1)
        return self

    def get_state(self, qubits='all'):
        if not self.n:
            return None
        qubits = self._check_qubit_arguments(qubits, False)
        self._reorder(qubits)
        if len(qubits) < self.n:
            return partial_trace(self.state.reshape(-1), qubits)
        return self.state

    def measure(self, qubits='all', obs=None):
        self._reset_unitary()
        qubits = self._check_qubit_arguments(qubits, False)
        if obs is not None:
            obs = self.parse_hermitian(obs, len(qubits))
            D, U = np.linalg.eig(obs)  # eig outputs a nicer basis than eigh
            self(U, qubits)  # basis change
        # play God
        probs = self._probs(qubits)
        choice = np.random.choice(2**len(qubits), p=probs)
        # collapse
        keep = self.state[choice]
        keep = normalize(keep)
        self.state = np.zeros_like(self.state)
        self.state[choice] = keep
        if obs is not None:
            self(U.T.conj(), qubits)  # basis change back to standard basis
        return binstr_from_int(choice, len(qubits))

    def ev(self, obs, qubits='all'):
        qubits = self._check_qubit_arguments(qubits, False)
        self._reorder(qubits)
        obs = self.parse_hermitian(obs, len(qubits))
        ev = self.state.conj().T @ obs @ self.state
        if len(qubits) < self.n:
            ev = np.trace(ev)
        return ev.real

    def std(self, obs, qubits='all', return_ev=False):
        ev = self.ev(obs, qubits)
        m2 = self.state.conj().T @ obs @ obs @ self.state
        if np.prod(m2.shape) > 1:
            m2 = np.trace(m2)
        var = m2 - ev**2
        std = np.sqrt(var.real)
        if return_ev:
            return std, ev
        return std

    def probs(self, qubits='all', obs=None):
        qubits = self._check_qubit_arguments(qubits, False)
        if obs is not None:
            obs = self.parse_hermitian(obs, len(qubits))
            D, U = np.linalg.eig(obs)
            self(U.T.conj(), qubits)  # basis change
            probs = self._probs(qubits)
            self(U, qubits)  # back to standard basis
            return probs
        return self._probs(qubits)

    def _probs(self, qubits='all'):
        self._reorder(qubits)
        probs = np.abs(self.state)**2
        if len(probs.shape) == 1:  # all qubits
            return probs
        return np.sum(probs, axis=1)

    def init(self, state, qubits=None):
        if qubits is None:
            if self.n == 0:  # infer `qubits` from `state`
                self.state = ket(state)
                n = count_qubits(self.state)
                self.qubits = list(range(n))
                self.original_order = list(range(n))
                self._reset_unitary()
                return self
            qubits = self.qubits
        else:
            qubits = self._check_qubit_arguments(qubits, True)

        new_state = ket(state, len(qubits))
        if len(qubits) < self.n:
            choice = self.measure(qubits)
            rest = self.state[int_from_binstr(choice)]
            self.state = np.kron(new_state, rest)
        else:
            self.state = new_state
        self._reset_unitary()
        return self

    def reset(self, qubits=None):
        if qubits is not None:
            self.init(0, qubits)
        elif self.n:
            self.init(0)
        # else: pass
        return self

    def random(self, n=None):
        n = n or self.n
        assert n, 'No qubits has been allocated yet'
        self.init(random_ket(n))
        return self

    def _alloc_qubit(self, q):
        if self._track_unitary:
            if len(self.state)**2*16*2 > psutil.virtual_memory().available:
                warnings.warn(f"RAM almost full ({self.n}-qubit unitary)")
        else:
            if len(self.state)*16*2 > psutil.virtual_memory().available:
                warnings.warn(f"RAM almost full ({self.n} qubit state)")

        self.qubits.append(q)
        self.original_order.append(q)
        self.state = np.kron(self.state.reshape(-1), [1,0]).reshape([2]*self.n)
        if self._track_unitary:
            self.U = np.kron(self.U, np.eye(2))

    def remove(self, qubits):
        qubits = self._check_qubit_arguments(qubits, False)
        if len(qubits) == self.n:
            return self.clear()
        choice = self.measure(qubits)
        self.state = self.state[int_from_binstr(choice)]
        if len(qubits) == self.n - 1:
            self.state = self.state.reshape([2])
        self.qubits = [q for q in self.qubits if q not in qubits]
        self.original_order = [q for q in self.original_order if q not in qubits]
        self._reset_unitary()
        return self

    def _check_qubit_arguments(self, qubits, allow_new):
        if isinstance(qubits, str) and qubits == 'all':
            qubits = self.original_order
        if not isinstance(qubits, (list, tuple, np.ndarray)):
            qubits = [qubits]
        qubits = list(qubits)
        assert len(qubits) > 0, "No qubits provided"
        for q in qubits:
            if q not in self.qubits:
                if allow_new:
                    self._alloc_qubit(q)
                else:
                    raise ValueError(f"Invalid qubit: {q}")
        assert len(set(qubits)) == len(qubits), f"Qubits should not contain a qubit multiple times, but was {qubits}"
        return qubits

    def _reset_unitary(self):
        if self._track_unitary:
            self.U = np.eye(2**self.n)

    @staticmethod
    def parse_unitary(U, n_qubits=None):
        if isinstance(U, (list, np.ndarray)):
            U = np.array(U)
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
        assert is_unitary(U), f"Unitary is not unitary: {U}"
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

    def _reorder(self, new_order):
        new_order_all = new_order + [q for q in self.qubits if q not in new_order]
        axes_new = [self.qubits.index(q) for q in new_order_all]
        self.qubits = new_order_all # update index dictionary with new locations
        if any(s > 2 for s in self.state.shape):
            self.state = self.state.reshape([2]*self.n)
        self.state = self.state.transpose(axes_new)
        if self._track_unitary:
            self.U = self.U.reshape([2,2]*self.n)
            U_axes_new = axes_new + [i + self.n for i in axes_new]
            self.U = self.U.transpose(U_axes_new)
        # collect
        n_front = len(new_order)
        if n_front < self.n:
            if self._track_unitary:
                self.U = self.U.reshape([2**n_front, 2**n_front, -1])
            self.state = self.state.reshape([2**n_front, -1])
        else:
            if self._track_unitary:
                self.U = self.U.reshape([2**n_front, 2**n_front])
            self.state = self.state.reshape(-1)

    def __getitem__(self, qubits):
        if isinstance(qubits, slice):
            indices = range(self.n)[qubits]
        else:
            if not hasattr(qubits, '__len__') or isinstance(qubits, str):
                qubits = [qubits]
            indices = [self.qubits.index(q) for q in qubits]
        return partial_trace(self.get_state(), indices)

    def __delitem__(self, qubits):
        if isinstance(qubits, slice):
            indices = range(self.n)[qubits]
            qubits = [self.qubits[idx] for idx in indices]
        elif not hasattr(qubits, '__len__') or isinstance(qubits, str):
            qubits = [qubits]
        self.remove(qubits)
        return self

    def plot(self, showqubits=None, showcoeff=True, showprobs=True, showrho=False, figsize=None, title=""):
        self._reorder(self.original_order)
        if showqubits is not None:
            self._check_qubit_arguments(showqubits, False)
            showqubits = [self.qubits.index(q) for q in showqubits]
        return plotQ(self.state, showqubits=showqubits, showcoeff=showcoeff, showprobs=showprobs, showrho=showrho, figsize=figsize, title=title)

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

    def qft(self, qubits):
        QFT = Fourier_matrix(n=len(qubits), n_is_qubits=True)
        return self(QFT, qubits)

    def iqft(self, qubits):
        QFT = Fourier_matrix(n=len(qubits), n_is_qubits=True)
        QFT_inv = QFT.T.conj()  # unitary!
        return self(QFT_inv, qubits)

    def pe(self, U, state, energy):
        # 1. Hadamard on energy register
        self.h(energy)

        # 2. Unitary condition
        for j, q in enumerate(energy[::-1]):  # [::-1] for big entian convention
            self.c(U**(2**j), q, state)

        # 3. IQFT on energy register
        self.iqft(energy)
        return self

    def __str__(self):
        state = self.get_state()
        if state is not None:
            state = unket(state)
        return f"qubits {self.original_order} in state '{state}'"

    def __repr__(self):
        return self.__str__()

qc = QuantumComputer()

## TODO: Implement number factoring using QuantumComputer (Shor's algorithm)

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

#############
### State ###
#############

def transpose_qubit_order(state, new_order):
    state = np.array(state)
    n = count_qubits(state)
    if new_order == -1:
        new_order = list(range(n)[::-1])

    new_order_all = new_order + [q for q in range(n) if q not in new_order]
    if len(state.shape) == 1:
        state = state.reshape([2]*n)
        state = state.transpose(new_order_all)
        state = state.reshape(-1)
    elif state.shape[0] == state.shape[1]:
        state = state.reshape([2,2]*n)
        new_order_all = new_order_all + [i + n for i in new_order_all]
        state = state.transpose(new_order_all)
        state = state.reshape([2**n, 2**n])
    else:
        raise ValueError(f"Not a valid shape: {state.shape}")
    return state

def reverse_qubit_order(state):
    """So the last will be first, and the first will be last."""
    return transpose_qubit_order(state, -1)

def partial_trace(rho, retain_qubits):
    """Trace out all qubits not specified in `retain_qubits`."""
    rho = np.array(rho)
    n = count_qubits(rho)

    # pre-process retain_qubits
    if is_int(retain_qubits):
        retain_qubits = [retain_qubits]
    dim_r = 2**len(retain_qubits)

    # get qubits to trace out
    trace_out = np.array(sorted(set(range(n)) - set(retain_qubits)))
    # ignore all qubits >= n
    trace_out = trace_out[trace_out < n]

    # if rho is a state vector
    if len(rho.shape) == 1:
        st  = rho.reshape([2]*n)
        rho = np.tensordot(st, st.conj(), axes=(trace_out,trace_out))
    # if trace out all qubits, just return the normal trace
    elif len(trace_out) == n:
        return np.trace(rho).reshape(1,1) # dim_r is not necessarily 1 here (if some in `retain_qubits` are >= n)
    else:
        assert rho.shape[0] == rho.shape[1], f"Can't trace a non-square matrix {rho.shape}"

        rho = rho.reshape([2]*(2*n))
        for qubit in trace_out:
            rho = np.trace(rho, axis1=qubit, axis2=qubit+n)
            n -= 1         # one qubit less
            trace_out -= 1 # rename the axes (only "higher" ones are left)

    # transpose the axes of the remaining qubits to the original order
    # rows = np.argsort(list(retain_qubits))
    # cols = rows + len(rows)
    # rho = rho.transpose(rows.tolist() + cols.tolist())

    return rho.reshape(dim_r, dim_r)

def state_trace(state, retain_qubits):
    """This is a pervert version of the partial trace, but for state vectors. I'm not sure about the physical 
    meaning of its output, but it was at times helpful to visualize and interpret subsystems, especially when 
    the density matrix was out of reach (or better: out of memory)."""
    state = np.array(state)
    state[np.isnan(state)] = 0
    n = count_qubits(state)

    # sanity checks
    if not hasattr(retain_qubits, '__len__'):
        retain_qubits = [retain_qubits]
    if len(retain_qubits) == 0:
        retain_qubits = range(n)
    elif n == len(retain_qubits):
        return state, np.abs(state)**2
    elif max(retain_qubits) >= n:
        raise ValueError(f"No such qubit: %d" % max(retain_qubits))

    state = state.reshape(tuple([2]*n))
    probs = np.abs(state)**2

    cur = 0
    for i in range(n):
        if i not in retain_qubits:
            state = np.sum(state, axis=cur)
            probs = np.sum(probs, axis=cur)
        else:
            cur += 1

    state = state.reshape(-1)
    state = normalize(state) # renormalize

    probs = probs.reshape(-1)
    assert np.abs(np.sum(probs) - 1) < 1e-5, np.sum(probs) # sanity check

    return state, probs

def plotQ(state, showqubits=None, showcoeff=True, showprobs=True, showrho=False, figsize=None, title=""):
    """My best attempt so far to visualize a state vector. Control with `showqubits` which subsystem you're 
    interested in (`None` will show the whole state). `showcoeff` utilitzes `state_trace`, `showprobs` shows 
    a pie chart of the probabilities when measured in the standard basis, and `showrho` gives a plt.imshow view 
    on the corresponding density matrix."""

    def tobin(n, places):
        return ("{0:0" + str(places) + "b}").format(n)

    def plotcoeff(ax, state):
        n = count_qubits(state)
        if n < 6:
            basis = [tobin(i, n) for i in range(2**n)]
            #plot(basis, state, ".", figsize=(10,3))
            ax.scatter(basis, state.real, label="real")
            ax.scatter(basis, state.imag, label="imag")
            ax.tick_params(axis="x", rotation=45)
        elif n < 9:
            ax.scatter(range(2**n), state.real, label="real")
            ax.scatter(range(2**n), state.imag, label="imag")
        else:
            ax.plot(range(2**n), state.real, label="real")
            ax.plot(range(2**n), state.imag, label="imag")

        #from matplotlib.ticker import StrMethodFormatter
        #ax.xaxis.set_major_formatter(StrMethodFormatter("{x:0"+str(n)+"b}"))
        ax.legend()
        ax.grid()

    def plotprobs(ax, state):
        n = count_qubits(state)
        toshow = {}
        cumsum = 0
        for idx in probs.argsort()[-20:][::-1]: # only look at 20 largest
            if cumsum > 0.96 or probs[idx] < 0.01:
                break
            toshow[tobin(idx, n)] = probs[idx]
            cumsum += probs[idx]
        if np.abs(1-cumsum) > 1e-15:
            toshow["rest"] = max(0,1-cumsum)
        ax.pie(toshow.values(), labels=toshow.keys(), autopct=lambda x: f"%.1f%%" % x)

    def plotrho(ax, rho):
        n = count_qubits(rho)
        rho = colorize_complex(rho)
        ax.imshow(rho)
        if n < 6:
            basis = [tobin(i, n) for i in range(2**n)]
            ax.set_xticks(range(rho.shape[0]), basis)
            ax.set_yticks(range(rho.shape[0]), basis)
            ax.tick_params(axis="x", rotation=45)

    state = np.array(state)

    # trace out unwanted qubits
    if showqubits is None:
        n = count_qubits(state)
        showqubits = range(n)

    if showrho:
        memory_requirement = (len(state))**2 * 16
        #print(memory_requirement / 1024**2, "MB") # rho.nbytes
        if memory_requirement > psutil.virtual_memory().available:
            raise ValueError(f"Too much memory required ({duh(memory_requirement)}) to calulate the density matrix!")
        rho = np.outer(state, state.conj())
        rho = partial_trace(rho, retain_qubits=showqubits)
    state, probs = state_trace(state, showqubits)

    if showcoeff and showprobs and showrho:
        if figsize is None:
            figsize=(16,4)
        fig, axs = plt.subplots(1,3, figsize=figsize)
        fig.subplots_adjust(right=1.2)
        plotrho(axs[0], rho)
        plotcoeff(axs[1], state)
        plotprobs(axs[2], state)
    elif showcoeff and showprobs:
        if figsize is None:
            figsize=(18,4)
        fig, axs = plt.subplots(1,2, figsize=figsize)
        fig.subplots_adjust(right=1.2)
        plotcoeff(axs[0], state)
        plotprobs(axs[1], state)
    elif showcoeff and showrho:
        if figsize is None:
            figsize=(16,4)
        fig, axs = plt.subplots(1,2, figsize=figsize)
        fig.subplots_adjust(right=1.2)
        plotcoeff(axs[0], state)
        plotrho(axs[1], rho)
    elif showprobs and showrho:
        if figsize is None:
            figsize=(6,4)
        fig, axs = plt.subplots(1,2, figsize=figsize)
        plotrho(axs[0], rho)
        plotprobs(axs[1], state)
    else:
        fig, ax = plt.subplots(1, figsize=figsize)
        if showcoeff:
            plotcoeff(ax, state)
        elif showprobs:
            plotprobs(ax, state)
        elif showrho:
            plotrho(ax, rho)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

def random_ket(n=1):
    """Generate a random state vector ($2^{n+1}-2$ degrees of freedom). Normalized and without global phase."""
    real = np.random.random(2**n)
    imag = np.random.random(2**n)
    return normalize(real + 1j*imag)

def random_dm(n=1, pure=False):
    """Generate a random density matrix ($2^{n+1}-1$ degrees of freedom). Normalized and without global phase."""
    if pure:
        state = random_ket(n)
        return np.outer(state, state.conj())
    else:
        probs = normalize(np.random.random(2**n), p=1)
        kets = normalize(random_vec((2**n, 2**n), complex=True))
        return kets @ np.diag(probs) @ kets.conj().T

def ket_from_int(d, n=None):
    if not n:
        n = int(np.ceil(np.log2(d+1))) or 1
    elif d >= 2**n:
        raise ValueError(f"A {n}-qubit state doesn't have basis state {d} (max is {2**n-1})")
    # return np.array(bincoll_from_int(2**d, 2**n)[::-1], dtype=float)
    res = np.zeros(2**n)
    res[d] = 1
    return res

def ket(specification, n=None):
    """Convert a string or dictionary of strings and weights to a state vector. The string can be a binary number 
    or a combination of binary numbers and weights. The weights will be normalized to 1."""
    # if a string is given, convert it to a dictionary
    if isinstance(specification, (np.ndarray, list, tuple)):
        n = n or count_qubits(specification) or 1
        assert len(specification) == 2**n, f"State vector has wrong length: {len(specification)} ≠ {2**n}!"
        return normalize(specification)
    if is_int(specification):
        return ket_from_int(specification, n)
    if type(specification) == str:
        if specification == "random" and n is not None:
            return random_ket(n)
        # handle some special cases: |+>, |->, |i>, |-i>
        if specification == "+":
            return normalize(np.array([1,1], dtype=complex))
        elif specification == "-":
            return normalize(np.array([1,-1], dtype=complex))
        elif specification == "i":
            return normalize(np.array([1,1j], dtype=complex))
        elif specification == "-i":
            return normalize(np.array([1,-1j], dtype=complex))

        # remove whitespace
        specification = specification.replace(" ", "")
        specification_dict = dict()

        # Parse the specification into the dictionary, where the keys are the strings '00', '01', '10', '11', etc. and the values are the weights
        # The following cases have to be considered:
        #  00 + 11
        #  00 - 11
        #  00 - 0.1*11
        #  0.5*00 + 0.5*11
        #  0.5*(00 + 11)
        #  (1+1j)*00 + (1-1j)*11
        #  0.5*((1+1j)*00 + (1-1j)*11)
        #  0.5*((1+1j)*00 + (1-1j)*11) + 0.5*((1-1j)*00 + (1+1j)*11)

        # if there are no brackets, then split by "+" and then by "*"
        if "(" not in specification and ")" not in specification:
            specification = specification.replace("-", "+-")
            for term in specification.split("+"):
                if term == "":
                    continue
                if "*" in term:
                    weight, state = term.split("*")
                    weight = complex(weight)
                elif term[0] == '-':
                    weight = -1
                    state = term[1:]
                else:
                    weight = 1
                    state = term
                if n is None:
                    n = len(state)
                else:
                    assert len(state) == n, f"Part of the specification has wrong length: len('{state}') != {n}"
                if state in specification_dict:
                    specification_dict[state] += weight
                else:
                    specification_dict[state] = weight

            specification = specification_dict
        else:
            raise NotImplementedError("Parentheses are not yet supported!")

    # convert the dictionary to a state vector
    n = len(list(specification.keys())[0])
    state = np.zeros(2**n, dtype=complex)
    for key in specification:
        state[int(key, 2)] = specification[key]
    return normalize(state)

def unket(state, as_dict=False, prec=5):
    """ Reverse of above. The output is always guaranteed to be normalized.

    `prec` serves as filter for states close to 0, and if `as_dict==False`, it also defines to which precision 
    the values are rounded in the string.

    Example:
    >>> unket(ket('00+01+10+11'))
    '0.5*(00+01+10+11)'
    >>> unket(ket('00+01+10+11'), as_dict=True)
    {'00': 0.5, '01': 0.5, '10': 0.5, '11': 0.5}
    """
    eps = 10**(-prec) if prec is not None else 0
    state = normalize(state)
    n = count_qubits(state)
    if as_dict:
        # cast to float if imaginary part is zero
        if np.allclose(state.imag, 0):
            state = state.real
        return {binstr_from_int(i, n): state[i] for i in range(2**n) if np.abs(state[i]) > eps}

    if prec is not None:
        state = np.round(state, prec)
    # group by weight
    weights = {}
    for i in range(2**n):
        if np.abs(state[i]) > eps:
            weight = state[i]
            pre = ''
            # Find a close value in weights
            for w in weights:
                if abs(w - weight) < eps:
                    weight = w
                    break
                if abs(w + weight) < eps:
                    weight = w
                    pre = '-'
                    break
            else:
                # remove imaginary part if it's zero
                if np.abs(weight.imag) < eps:
                    weight = weight.real
                # create new weight
                weights[weight] = []
            # add state to weight
            weights[weight].append(pre + binstr_from_int(i, n))

    # convert to string
    res = []
    for weight in weights:
        if weight == 1:
            res += weights[weight]
        elif len(weights[weight]) == 1:
            res += [f"{weight}*{weights[weight][0]}"]
        else:
            res += [f"{weight}*({'+'.join(weights[weight])})"]
    return "+".join(res).replace("+-", "-")

def op(specification1, specification2=None, n=None):
    # If it's already a matrix, ensure it's a density matrix and return it
    if isinstance(specification1, (list, np.ndarray)):
        specification1 = np.array(specification1, copy=False)
        if len(specification1.shape) > 1:
            sp1_trace = np.trace(specification1)
            # trace normalize it if it's not already
            if not abs(sp1_trace - 1) < 1e-8:
                specification1 = specification1 / sp1_trace
            return specification1
    s1 = ket(specification1, n)
    s2 = s1 if specification2 is None else ket(specification2, count_qubits(s1))
    return np.outer(s1, s2.conj())

def dm(specification1, specification2=None):
    rho = op(specification1, specification2)
    assert is_dm(rho), f"The given matrix is not a density matrix!"
    return rho

def ev(observable, psi):
    # assert is_hermitian(observable)
    return (psi.conj() @ observable @ psi).real

def probs(state):
    """Calculate the probabilities of measuring a state vector in the standard basis."""
    return np.abs(state)**2

def is_dm(rho):
    """Check if matrix `rho` is a density matrix."""
    rho = np.array(rho)
    n = count_qubits(rho)
    if len(rho.shape) != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] != 2**n:
        return False
    return abs(np.trace(rho) - 1) < 1e-10 and is_psd(rho)

def is_pure_dm(rho):
    """Check if matrix `rho` is a pure density matrix."""
    if not is_dm(rho):
        return False
    # return np.linalg.matrix_rank(rho) == 1
    return abs(np.trace(rho @ rho) - 1) < 1e-10

def is_eigenstate(psi, H):
    psi = normalize(psi)
    psi2 = normalize(H @ psi)
    return abs(abs(psi2 @ psi) - 1) < 1e-10

def gibbs(H, beta=1):
    """Calculate the Gibbs state of a Hamiltonian `H` at inverse temperature `beta`."""
    H = np.array(H)
    assert is_hermitian(H), "Hamiltonian must be Hermitian!"
    assert beta >= 0, "Inverse temperature must be positive!"
    E, U = np.linalg.eigh(H)
    E = softmax(E, -beta)
    return U @ np.diag(E) @ U.conj().T

def count_qubits(obj):
    if hasattr(obj, '__len__') and not isinstance(obj, str):
        n = int(np.log2(len(obj)))
        assert len(obj) == 2**n, f"Dimension must be a power of 2, but was {len(obj)}"
        return n
    if isinstance(obj, str):
        import re
        obj = obj.replace(' ', '')  # remove spaces
        if "+" in obj or "-" in obj:
            if 'X' in obj or 'Y' in obj or 'Z' in obj or 'I' in obj:
                # obj = parse_hamiltonian(obj)
                return len(re.search('[XYZI]+', obj)[0])
            else:
                # obj = ket(obj)
                return len(re.search('[01]+(?![.*01])', obj)[0])
        else:
            # obj = parse_unitary(obj)
            obj = obj.split('@')[0]
            return len(re.search('\\S+', obj)[0])
    if hasattr(obj, 'n'):
        return obj.n
    if hasattr(obj, 'num_qubits'):
        return obj.num_qubits
    if hasattr(obj, 'qubits'):
        return len(obj.qubits)
    raise ValueError(f'Unkown object: {obj}')

##################################
### Quantum information theory ###
##################################

def entropy_von_Neumann(state):
    """Calculate the von Neumann entropy of a given density matrix."""
    if not(isinstance(state, np.ndarray) and len(state.shape) == 2 and state.shape[0] == state.shape[1]):
        state = op(state)
    # S = -np.trace(state @ matlog(state)/np.log(2))
    # assert np.allclose(S.imag, 0), f"WTF: Entropy is not real: {S}"
    # return np.max(S.real, 0)  # fix rounding errors
    eigs = np.linalg.eigvalsh(state)
    assert abs(np.sum(eigs) - 1) < 1e-10, f"Density matrix is not normalized! {np.sum(eigs)}"
    assert np.all(eigs >= -len(eigs)*sys.float_info.epsilon), f"Density matrix is not positive semidefinite! {eigs}"
    return entropy(eigs)

def entropy_entanglement(state, subsystem_qubits):
    """Calculate the entanglement entropy of a quantum state (density matrix or vector) with respect to the given subsystem."""
    return entropy_von_Neumann(partial_trace(state, subsystem_qubits))

def mutual_information_quantum(state, subsystem_qubits):
    n = count_qubits(state)
    rho_A = partial_trace(state, subsystem_qubits)
    rho_B = partial_trace(state, [s for s in range(n) if s not in subsystem_qubits])
    return entropy_von_Neumann(rho_A) + entropy_von_Neumann(rho_B) - entropy_von_Neumann(state)

def fidelity(state1, state2):
    """Calculate the fidelity between two quantum states."""
    state1 = np.array(state1)
    state2 = np.array(state2)

    if len(state1.shape) == 1 and len(state2.shape) == 1:
        return np.abs(state1 @ state2.conj())**2
    elif len(state1.shape) == 2 and len(state2.shape) == 1:
        return np.abs(state2.conj() @ state1 @ state2)
    elif len(state1.shape) == 1 and len(state2.shape) == 2:
        return np.abs(state1.conj() @ state2 @ state1)
    elif len(state1.shape) == 2 and len(state2.shape) == 2:
        # state1_sqrt = matsqrt(state1)
        # return np.trace(matsqrt(state1_sqrt @ state2 @ state1_sqrt))**2
        return np.sum(np.sqrt(np.linalg.eigvals(state1 @ state2)))**2 # this is correct and faster
    else:
        raise ValueError(f"Can't calculate fidelity between {state1.shape} and {state2.shape}")

def trace_distance(rho1, rho2):
    """Calculate the trace distance between two density matrices."""
    rho1, rho2 = op(rho1), op(rho2)
    return 0.5 * trace_norm(rho1 - rho2)

def Schmidt_decomposition(state, subsystem_qubits):
    """Calculate the Schmidt decomposition of a pure state with respect to the given subsystem."""
    state = np.array(state)
    assert len(state.shape) == 1, f"State must be a vector, but has shape: {state.shape}"
    n = int(np.log2(len(state)))
    assert len(subsystem_qubits) <= n-1, f"Too many subsystem qubits: {len(subsystem_qubits)} >= {n}"

    # reorder the qubits so that the subsystem qubits are at the beginning
    subsystem_qubits = list(subsystem_qubits)
    other_qubits = sorted(set(range(n)) - set(subsystem_qubits))
    state = state.reshape([2]*n).transpose(subsystem_qubits + other_qubits).reshape(-1)

    # calculate the Schmidt coefficients and basis using SVD
    a_jk = state.reshape([2**len(subsystem_qubits), 2**len(other_qubits)])
    U, S, V = np.linalg.svd(a_jk)
    return S, U.T, V

def correlation_quantum(state, observable_A, observable_B):
    n_A = int(np.log2(len(observable_A)))
    n_B = int(np.log2(len(observable_B)))
    state = np.array(state)
    if len(state.shape) == 1:
        state = ket(state)
        observable_AB = np.kron(observable_A, observable_B)
        observable_AI = np.kron(observable_A, np.eye(2**n_B))
        observable_IB = np.kron(np.eye(2**n_A), observable_B)
        return ev(observable_AB, state) - ev(observable_AI, state)*ev(observable_IB, state)
    else:
        state = dm(state)
        rho_A = partial_trace(state, list(range(n_A)))
        rho_B = partial_trace(state, list(range(n_A, n_A + n_B)))
        observable_AB = np.kron(observable_A, observable_B)
        return np.trace(observable_AB @ (state - np.kron(rho_A, rho_B))).real


####################
### Ground state ###
####################

def ground_state_exact(hamiltonian):
    """ Calculate the ground state using exact diagonalization. """
    if isinstance(hamiltonian, str):
        H = parse_hamiltonian(hamiltonian, sparse=True)
    # elif isinstance(hamiltonian, nk.operator.DiscreteOperator):
    #     H = hamiltonian.to_sparse()
    else:
        H = hamiltonian

    if sp.issparse(H):
        evals, evecs = sp.linalg.eigsh(H, k=1, which='SA')
    else:
        evals, evecs = np.linalg.eigh(H) # way faster, but only dense matrices
    ground_state_energy = evals[0]
    ground_state = evecs[:,0]
    return ground_state_energy, ground_state

def get_E0(H):
    return ground_state_exact(H)[0]

def ground_state_ITE(H, tau=5, eps=1e-6):  # eps=1e-6 gives almost perfect precision in the energy
    """ Calculate the ground state using the Imaginary Time-Evolution (ITE) scheme.
    Since its vanilla form uses diagonalization (to calculate the matrix exponential), it can't be more efficient than diagonalization itself. """
    def evolve(i, psi):
        psi = U @ psi
        return normalize(psi)

    # U = matexp(-tau*H)
    D, V = np.linalg.eigh(H)  # this requires diagonalization of H
    U = V @ np.diag(softmax(D, -tau)) @ V.conj().T
    n = int(np.log2(H.shape[0]))
    ground_state = sequence(evolve, start_value=random_ket(n), eps=eps)
    ground_state_energy = (ground_state.conj().T @ H @ ground_state).real
    return ground_state_energy, ground_state

def power_iteration(A, eps=1e-8):
    """ Power iteration algorithm. If A has a real, positive, unique eigenvalue of largest magnitude, this outputs it and the associated eigenvector."""
    eigvec = sequence(
        lambda i, b: normalize(A @ b),
        start_value=random_vec(A.shape[1]),
        eps=eps
    )
    eigval = eigvec.T.conj() @ A @ eigvec
    return eigval, eigvec

###################
### Hamiltonian ###
###################

matmap_np, matmap_sp = None, None

def parse_hamiltonian(hamiltonian, sparse=False, scaling=1, buffer=None, max_buffer_n=0, dtype=complex):
    """Parse a string representation of a Hamiltonian into a matrix representation. The result is guaranteed to be Hermitian.
    The string `hamiltonian` must follow the following syntax (see examples below)
        ```
        <hamiltonian> = <term> | <term> + <hamiltonian>
        <term>        = <pauli> | (<hamiltonian>) | <weight>*<pauli> | <weight>*(<hamiltonian>) | <weight>
        <pauli>       = [IXYZ]+
        ```
    where `<weight>` is a string representing a finite number that can be parsed by `float()`. If `<weight>` is not followed by `*`, it is assumed to be the respective multiple of the identity.

    Parameters:
        hamiltonian (str): The Hamiltonian to parse.
        sparse (bool): Whether to use sparse matrices (csr_matrix) or dense matrices (numpy.array).
        scaling (float): A constant factor to scale the Hamiltonian by.
        buffer (dict): A dictionary to store calculated chunks in. If `None`, it defaults to the global `matmap_np` (or `matmap_sp` if `sparse == True`). Give `buffer={}` and leave `max_buffer_n == 0` (default) to disable the buffer.
        max_buffer_n (int): The maximum length (number of qubits) for new chunks to store in the buffer (default: 0). If `0`, no new chunks will be stored in the buffer.

    Returns:
        numpy.ndarray | scipy.sparse.csr_matrix: The matrix representation of the Hamiltonian.

    Example:
    >>> parse_hamiltonian('0.5*(XX + YY + ZZ + II)') # SWAP
    array([[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
           [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
           [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
           [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]])
    >>> parse_hamiltonian('X - 2*Y + 1')
    array([[1.+0.j, 1.+2.j],
           [1.-2.j, 1.+0.j]])
    >>> parse_hamiltonian('-(XX + YY + .5*ZZ) + 1.5')
    array([[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
           [ 0.+0.j  2.+0.j -2.+0.j  0.+0.j]
           [ 0.+0.j -2.+0.j  2.+0.j  0.+0.j]
           [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]])
    >>> parse_hamiltonian('0.5*(II + ZI - ZX + IX)') # CNOT

    """
    kron = sp.kron if sparse else np.kron

    # Initialize the matrix map
    global matmap_np, matmap_sp
    if matmap_np is None or matmap_sp is None or matmap_np["I"].dtype != dtype:
        # numpy versions
        matmap_np = {
            "H": np.array([[1, 1], [1, -1]], dtype=dtype) / np.sqrt(2),
            "X": np.array([[0, 1], [1, 0]], dtype=dtype),
            "Z": np.array([[1, 0], [0, -1]], dtype=dtype),
            "I": np.array([[1, 0], [0, 1]], dtype=dtype),
        }
        # composites
        matmap_np.update({
            "ZZ": np.kron(matmap_np['Z'], matmap_np['Z']),
            "IX": np.kron(matmap_np['I'], matmap_np['X']),
            "XI": np.kron(matmap_np['X'], matmap_np['I']),
            "YY": np.array([[ 0,  0,  0, -1],  # to avoid complex numbers
                            [ 0,  0,  1,  0],
                            [ 0,  1,  0,  0],
                            [-1,  0,  0,  0]], dtype=dtype)
        })
        for i in range(2, 11):
            matmap_np["I"*i] = np.eye(2**i, dtype=dtype)
        # add 'Y' only if dtype supports imaginary numbers
        if np.issubdtype(dtype, np.complexfloating):
            matmap_np["Y"] = np.array([[0, -1j], [1j, 0]], dtype=dtype)

        # sparse versions
        matmap_sp = {k: sp.csr_array(v) for k, v in matmap_np.items()}

    if not np.issubdtype(dtype, np.complexfloating) and "Y" in hamiltonian:
        raise ValueError(f"The Pauli matrix Y is not supported for dtype {dtype.__name__}.")

    matmap = matmap_sp if sparse else matmap_np

    # only use buffer if pre-computed chunks are available or if new chunks are allowed to be stored
    use_buffer = buffer is None or len(buffer) > 0 or max_buffer_n > 0
    if use_buffer and buffer is None:
        buffer = matmap

    def calculate_chunk_matrix(chunk, sparse=False, scaling=1):
        # if scaling != 1:  # only relevant for int dtype, to prevent changing dtype when multiplying
            # scaling = np.array(scaling, dtype=dtype)
        if use_buffer:
            if chunk in buffer:
                return buffer[chunk] if scaling == 1 else scaling * buffer[chunk]
            if len(chunk) == 1:
                return matmap[chunk[0]] if scaling == 1 else scaling * matmap[chunk[0]]
            # Check if a part of the chunk has already been calculated
            for i in range(len(chunk)-1, 1, -1):
                for j in range(len(chunk)-i+1):
                    subchunk = chunk[j:j+i]
                    if subchunk in buffer:
                        # If so, calculate the rest of the chunk recursively
                        parts = [chunk[:j], subchunk, chunk[j+i:]]
                        # remove empty chunks
                        parts = [c for c in parts if c != ""]
                        # See where to apply the scaling
                        shortest = min(parts, key=len)
                        # Calculate each part recursively
                        for i, c in enumerate(parts):
                            if c == subchunk:
                                if c == shortest:
                                    parts[i] = scaling * buffer[c]
                                    shortest = ""
                                else:
                                    parts[i] = buffer[c]
                            else:
                                if c == shortest:
                                    parts[i] = calculate_chunk_matrix(c, sparse=sparse, scaling=scaling)
                                    shortest = ""
                                else:
                                    parts[i] = calculate_chunk_matrix(c, sparse=sparse, scaling=1)
                        return reduce(kron, parts)

        # Calculate the chunk matrix gate by gate
        if use_buffer and len(chunk) <= max_buffer_n:
            gates = [matmap[gate] for gate in chunk]
            chunk_matrix = reduce(kron, gates)
            buffer[chunk] = chunk_matrix
            if scaling != 1:
                chunk_matrix = scaling * chunk_matrix
        else:
            gates = [scaling * matmap[chunk[0]]] + [matmap[gate] for gate in chunk[1:]]
            chunk_matrix = reduce(kron, gates)

        return chunk_matrix

    # Remove whitespace
    hamiltonian = hamiltonian.replace(" ", "")
    # replace - with +-, except before e
    hamiltonian = hamiltonian \
                    .replace("-", "+-") \
                    .replace("e+-", "e-") \
                    .replace("(+-", "(-")

    # print("ph: Pre-processed Hamiltonian:", hamiltonian)

    # Find parts in parentheses
    part = ""
    parts = []
    depth = 0
    current_part_weight = ""
    for i, c in enumerate(hamiltonian):
        if c == "(":
            if depth == 0:
                # for top-level parts search backwards for the weight
                weight = ""
                for j in range(i-1, -1, -1):
                    if hamiltonian[j] in ["("]:
                        break
                    weight += hamiltonian[j]
                    if hamiltonian[j] in ["+", "-"]:
                        break
                weight = weight[::-1]
                if weight != "":
                    current_part_weight = weight
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                part += c
                parts.append((current_part_weight, part))
                part = ""
                current_part_weight = ""
        if depth > 0: 
            part += c

    # print("Parts found:", parts)

    # Replace parts in parentheses with a placeholder
    for i, (weight, part) in enumerate(parts):
        hamiltonian = hamiltonian.replace(weight+part, f"+part{i}", 1)
        # remove * at the end of the weight
        if weight != "" and weight[-1] == "*":
            weight = weight[:-1]
        if weight in ["", "+", "-"]:
            weight += "1"
        # Calculate the part recursively
        part = part[1:-1] # remove parentheses
        parts[i] = parse_hamiltonian(part, sparse=sparse, scaling=float(weight), buffer=buffer, max_buffer_n=max_buffer_n, dtype=dtype)

    # print("Parts replaced:", parts)

    # Parse the rest of the Hamiltonian
    chunks = hamiltonian.split("+")
    # Remove empty chunks
    chunks = [c for c in chunks if c != ""]
    # If parts are present, use them to determine the number of qubits
    if parts:
        n = int(np.log2(parts[0].shape[0]))
    else: # Use chunks to determine the number of qubits
        n = 0
        for c in chunks:
            if c[0] in ["-", "+"]:
                c = c[1:]
            if "*" in c:
                c = c.split("*")[1]
            if c.startswith("part"):
                continue
            try:
                float(c)
                continue
            except ValueError:
                n = len(c)
                break
        if n == 0:
            warnings.warn("Hamiltonian is a scalar!")

    if not sparse and n > 10:
        # check if we would blow up the memory
        mem_required = 2**(2*n) * np.array(1, dtype=dtype).nbytes
        mem_available = psutil.virtual_memory().available
        if mem_required > mem_available:
            raise MemoryError(f"This would blow up you memory ({duh(mem_required)} required)! Try using `sparse=True`.")

    if sparse:
        H = sp.csr_array((2**n, 2**n), dtype=dtype)
    else:
        if n > 10:
            warnings.warn(f"Using a dense matrix for a {n}-qubit Hamiltonian is not recommended. Use sparse=True.")
        H = np.zeros((2**n, 2**n), dtype=dtype)

    for chunk in chunks:
        # print("Processing chunk:", chunk)
        if chunk == "":
            continue
        chunk_matrix = None
        # Parse the weight of the chunk
        if chunk.startswith("part"):
            weight = 1  # parts already include their weight
            chunk_matrix = parts[int(chunk.split("part")[1])]
        elif "*" in chunk:
            weight = float(chunk.split("*")[0])
            chunk = chunk.split("*")[1]
        elif len(chunk) == n+1 and chunk[0] in ["-", "+"] and n >= 1 and chunk[1] in matmap:
            weight = float(chunk[0] + "1")
            chunk = chunk[1:]
        elif (chunk[0] in ["-", "+", "."] or chunk[0].isdigit()) and all([c not in matmap for c in chunk[1:]]):
            if len(chunk) == 1 and chunk[0] in ["-", "."]:
                chunk = 0
            weight = complex(chunk)
            if np.iscomplex(weight):
                raise ValueError("Complex scalars would make the Hamiltonian non-Hermitian!")
            weight = weight.real
            # weight = np.array(weight, dtype=dtype)  # only relevant for int dtype
            chunk_matrix = np.eye(2**n, dtype=dtype)
        elif len(chunk) != n:
            raise ValueError(f"Gate count must be {n} but was {len(chunk)} for chunk \"{chunk}\"")
        else:
            weight = 1

        if chunk_matrix is None:
            chunk_matrix = calculate_chunk_matrix(chunk, sparse=sparse, scaling=scaling * weight)
        elif scaling * weight != 1:
            chunk_matrix = scaling * weight * chunk_matrix

        # Add the chunk to the Hamiltonian
        # print("Adding chunk", weight, chunk, "for hamiltonian", scaling, hamiltonian)
        # print(type(H), H.dtype, type(chunk_matrix), chunk_matrix.dtype)
        if len(chunks) == 1:
            H = chunk_matrix
        else:
            H += chunk_matrix

    if sparse:
        assert np.allclose(H.data, H.conj().T.data), f"The given Hamiltonian {hamiltonian} is not Hermitian: {H.data}"
    else:
        assert is_hermitian(H), f"The given Hamiltonian {hamiltonian} is not Hermitian: {H}"

    return H

def random_ham(n_qubits, n_terms, offset=0, coeffs=True, scaling=True):
    """ Draw `n_terms` basis elements out of the basis. If `scaling=True`, the coefficients are normalized to 1. If `coeffs=False`, the coefficients (and scaling) are omitted."""
    # generate a list of random terms
    basis = pauli_basis(n_qubits, kind='str')[1:]  # exclude the identity
    assert n_terms <= len(basis), f"Can't draw {n_terms} terms from a basis of size {len(basis)}!"
    terms = np.random.choice(basis, size=n_terms, replace=False)
    if not coeffs:
        return ' + '.join(terms)
    # generate a random coefficient for each term
    coeffs = np.random.random(n_terms)
    if scaling:
        coeffs = normalize(coeffs, p=1)
    # generate the Hamiltonian
    H_str = ' + '.join([f'{coeffs[i]}*{term}' for i, term in enumerate(terms)])
    if offset != 0:
        H_str += ' + ' + str(offset)
    return H_str

def ising(n_qubits, J=(-1,1), h=(-1,1), g=(-1,1), offset=0, kind='1d', circular=False, as_dict=False, prec=42):
    """
    Generates an Ising model with (optional) longitudinal and (optional) transverse couplings.

    Parameters
    ----------
    n_qubits : int or tuple
        Number of qubits, at least 2. For `kind='2d'` or `kind='3d'`, give a tuple of 2 or 3 integers, respectively.
    J : float, array, or dict
        Coupling strength.
        If a scalar, all couplings are set to this value.
        If a 2-element vector but `n_qubit > 2` (or tuple), all couplings are set to a random value in this range.
        If a matrix, this matrix is used as the coupling matrix.
        For `kind='pairwise'`, `kind='2d'`, or `kind='3d'`, `J` is read as an incidence matrix, where the rows and columns correspond to the qubits and the values are the coupling strengths. Only the upper triangular part of the matrix is used.
        For `kind='all'`, specify a dictionary with keys being tuples of qubit indices and values being the corresponding coupling strength.
    h : float or array, optional
        Longitudinal field strength.
        If a scalar, all fields are set to this value.
        If a 2-element vector, all fields are set to a random value in this range.
        If a vector of size `n_qubits`, its elements specify the individual strengths of the longitudinal field.
    g : float or array, optional
        Transverse field strength.
        If a scalar, all couplings are set to this value.
        If a 2-element vector, all couplings are set to a random value in this range.
        If a vector of size `n_qubits`, its elements specify the individual strengths of the transverse field.
    offset : float, optional
        Offset of the Hamiltonian.
    kind : {'1d', '2d', '3d', 'pairwise', 'all'}, optional
        Whether the couplings are along a string (`1d`), on a 2d-lattice (`2d`), 3d-lattice (`3d`), fully connected graph (`pairwise`), or specify the desired multi-particle interactions.
    circular : bool, optional
        Whether the couplings are circular (i.e. the outermost qubits are coupled to each other). Only applies to `kind='1d'`, `kind='2d'`, and `kind='3d'`.

    Returns
    -------
    H : str
        The Hamiltonian as a string, which can be parsed by ph.
    """
    # TODO: interactions on X and Y, and local fields on Y
    # if n_qubits is not scalar or tuple, try ising_graph
    if not (np.isscalar(n_qubits) or isinstance(n_qubits, tuple)):
        return ising_graph(n_qubits, J=J, h=h, g=g, offset=offset)
    # generate the coupling shape
    n_total_qubits = np.prod(n_qubits)
    if n_total_qubits < 1:
        raise ValueError(f"Number of qubits must be positive, but is {n_total_qubits}")
    assert n_total_qubits - int(n_total_qubits) == 0, "n_qubits must be an integer or a tuple of integers"
    if kind == '1d':
        assert np.isscalar(n_qubits) or len(n_qubits) == 1, f"For kind={kind}, n_qubits must be an integer or tuple of length 1, but is {n_qubits}"
        # convert to int if tuple (has attr __len__)
        if hasattr(n_qubits, '__len__'):
            n_qubits = n_qubits[0]
        couplings = (n_qubits if circular and n_qubits > 2 else n_qubits-1,)
    elif kind == '2d':
        if np.isscalar(n_qubits) or len(n_qubits) != 2:
            raise ValueError(f"For kind={kind}, n_qubits must be a tuple of length 2, but is {n_qubits} ({type(n_qubits)})")
        couplings = (n_total_qubits, n_total_qubits)
    elif kind == '3d':
        if np.isscalar(n_qubits) or len(n_qubits) != 3:
            raise ValueError(f"For kind={kind}, n_qubits must be a tuple of length 3, but is {n_qubits} ({type(n_qubits)})")
        couplings = (n_total_qubits, n_total_qubits)
    elif kind == 'pairwise':
        assert type(n_qubits) == int, f"For kind={kind}, n_qubits must be an integer, but is {n_qubits} ({type(n_qubits)})"
        couplings = (n_qubits, n_qubits)
    elif kind == 'all':
        assert type(n_qubits) == int, f"For kind={kind}, n_qubits must be an integer, but is {n_qubits} ({type(n_qubits)})"
        couplings = (2**n_qubits,)
    else:
        raise ValueError(f"Unknown kind {kind}")

    # if J is not scalar or dict, it must be either the array of the couplings or the limits of the random range
    if not (np.isscalar(J) or isinstance(J, dict)):
        J = np.array(J)
        if J.shape == (2,):
            J = np.random.uniform(J[0], J[1], couplings)
        if kind == '1d' and J.shape == (n_qubits, n_qubits):
            # get the offset k=1 diagonal (n_qubits-1 elements)
            idxs = np.where(np.eye(n_qubits, k=1))
            if circular:
                # add the edge element
                idxs = (np.append(idxs[0], 0), np.append(idxs[1], n_qubits-1))
            J = J[idxs]
        if kind == 'pairwise' and len(J.shape) == 1 and len(J) == len(np.triu_indices(n_qubits, k=1)[0]):
            J_mat = np.zeros(couplings)
            J_mat[np.triu_indices(n_qubits, k=1)] = J
            J = J_mat
        assert J.shape == couplings, f"For kind={kind}, J must be a scalar, 2-element vector, or matrix of shape {couplings}, but is {J.shape}"
    elif isinstance(J, dict) and kind != 'all':
        raise ValueError(f"For kind={kind}, J must not be a dict!")

    if h is not None:
        if n_total_qubits != 2 and hasattr(h, '__len__') and len(h) == 2:
            h = np.random.uniform(low=h[0], high=h[1], size=n_total_qubits)
        elif not np.isscalar(h):
            h = np.array(h)
        assert np.isscalar(h) or h.shape == (n_total_qubits,), f"h must be a scalar, 2-element vector, or vector of shape {(n_total_qubits,)}, but is {h.shape if not np.isscalar(h) else h}"
    if g is not None:
        if n_total_qubits != 2 and hasattr(g, '__len__') and len(g) == 2:
            g = np.random.uniform(low=g[0], high=g[1], size=n_total_qubits)
        elif not np.isscalar(g):
            g = np.array(g)
        assert np.isscalar(g) or g.shape == (n_total_qubits,), f"g must be a scalar, 2-element vector, or vector of shape {(n_total_qubits,)}, but is {g.shape if not np.isscalar(g) else g}"

    # round number to desired precision
    if prec is not None:
        if isinstance(J, dict):
            J = {k: np.round(v, prec) for k, v in J.items()}
        else:
            J = np.round(J, prec)
        h = np.round(h, prec)
        g = np.round(g, prec)

    # generate the Hamiltonian
    H = {}
    # pairwise interactions
    if kind == '1d':
        if np.isscalar(J):
            J = [J]*n_qubits
        for i in range(n_qubits-1):
            if J[i] != 0:
                H['I'*i + 'ZZ' + 'I'*(n_qubits-i-2)] = J[i]
        # last and first qubit
        if circular and n_qubits > 2 and J[-1] != 0:
            H['Z' + 'I'*(n_qubits-2) + 'Z'] = J[-1]
    elif kind == '2d':
        for i in range(n_qubits[0]):
            for j in range(n_qubits[1]):
                # find all 2d neighbors, but avoid double counting
                neighbors = []
                if i > 0:
                    neighbors.append((i-1, j))
                if i < n_qubits[0]-1:
                    neighbors.append((i+1, j))
                if j > 0:
                    neighbors.append((i, j-1))
                if j < n_qubits[1]-1:
                    neighbors.append((i, j+1))
                if circular:
                    if i == n_qubits[0]-1 and n_qubits[0] > 2:
                        neighbors.append((0, j))
                    if j == n_qubits[1]-1 and n_qubits[1] > 2:
                        neighbors.append((i, 0))
                # add interactions
                index_node = i*n_qubits[1] + j
                for neighbor in neighbors:
                    # 1. lower row
                    # 2. same row, but further to the right or row circular (= first column and j is last column)
                    # 3. same column, but column circular (= first row and i is last row)
                    if neighbor[0] > i \
                        or (neighbor[0] == i and (neighbor[1] > j or (j == n_qubits[1]-1 and neighbor[1] == 0 and n_qubits[1] > 2))) \
                        or (neighbor[1] == j and i == n_qubits[0]-1 and neighbor[0] == 0 and n_qubits[0] > 2):
                        index_neighbor = neighbor[0]*n_qubits[1] + neighbor[1]
                        idx1 = min(index_node, index_neighbor)
                        idx2 = max(index_node, index_neighbor)
                        J_val = J[idx1, idx2] if not np.isscalar(J) else J
                        if J_val != 0:
                            H['I'*idx1 + 'Z' + 'I'*(idx2-idx1-1) + 'Z' + 'I'*(n_qubits[0]*n_qubits[1]-idx2-1)] = J_val
    elif kind == '3d':
        for i in range(n_qubits[0]):
            for j in range(n_qubits[1]):
                for k in range(n_qubits[2]):
                    # find all 3d neighbors, but avoid double counting
                    neighbors = []
                    if i > 0:
                        neighbors.append((i-1, j, k))
                    if i < n_qubits[0]-1:
                        neighbors.append((i+1, j, k))
                    if j > 0:
                        neighbors.append((i, j-1, k))
                    if j < n_qubits[1]-1:
                        neighbors.append((i, j+1, k))
                    if k > 0:
                        neighbors.append((i, j, k-1))
                    if k < n_qubits[2]-1:
                        neighbors.append((i, j, k+1))
                    if circular:
                        if i == n_qubits[0]-1 and n_qubits[0] > 2:
                            neighbors.append((0, j, k))
                        if j == n_qubits[1]-1 and n_qubits[1] > 2:
                            neighbors.append((i, 0, k))
                        if k == n_qubits[2]-1 and n_qubits[2] > 2:
                            neighbors.append((i, j, 0))
                    # add interactions
                    index_node = i*n_qubits[1]*n_qubits[2] + j*n_qubits[2] + k
                    for neighbor in neighbors:
                        # 1. lower row
                        # 2. same row, but
                            # a. same layer, but further to the right or row circular (= first column and j is last column)
                            # b. same column, but further behind or layer circular (= first layer and k is last layer)
                        # 3. same column and same layer, but column circular (= first row and i is last row)
                        if neighbor[0] > i \
                            or (neighbor[0] == i and (\
                                (neighbor[2] == k and (neighbor[1] > j or (j == n_qubits[1]-1 and neighbor[1] == 0 and n_qubits[1] > 2))) \
                                or (neighbor[1] == j and (neighbor[2] > k or (k == n_qubits[2]-1 and neighbor[2] == 0 and n_qubits[2] > 2))) \
                            )) \
                            or (neighbor[1] == j and neighbor[2] == k and i == n_qubits[0]-1 and neighbor[0] == 0 and n_qubits[0] > 2):
                            index_neighbor = neighbor[0]*n_qubits[1]*n_qubits[2] + neighbor[1]*n_qubits[2] + neighbor[2]
                            idx1 = min(index_node, index_neighbor)
                            idx2 = max(index_node, index_neighbor)
                            J_val = J[idx1, idx2] if not np.isscalar(J) else J
                            if J_val != 0:
                                H['I'*idx1 + 'Z' + 'I'*(idx2-idx1-1) + 'Z' + 'I'*(n_qubits[0]*n_qubits[1]*n_qubits[2]-idx2-1)] = J_val
    elif kind == 'pairwise':
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                J_val = J[i,j] if not np.isscalar(J) else J
                if J_val != 0:
                    H['I'*i + 'Z' + 'I'*(j-i-1) + 'Z' + 'I'*(n_qubits-j-1)] = J_val
    elif kind == 'all':
        if np.isscalar(J):
            if J == 0:
                raise ValueError("For kind='all', J must not be 0!")
            if n_qubits > 20:
                raise ValueError("Printing out all interactions for n_qubits > 20 is not recommended.")
            for i in range(2, n_qubits+1):
                for membership in itertools.combinations(range(n_qubits), i):
                    H[''.join('Z' if j in membership else 'I' for j in range(n_qubits))] = J
        else: # J is a dict of tuples of qubit indices to interaction strengths
            for membership, strength in J.items():
                if strength != 0:
                    H[''.join('Z' if j in membership else 'I' for j in range(n_qubits))] = strength
    else:
        raise ValueError(f"Unknown kind {kind}")

    # local longitudinal fields
    if np.any(h):
        for i in range(n_total_qubits):
            h_val = h[i] if not np.isscalar(h) else h
            if h_val != 0:
                H['I'*i + 'Z' + 'I'*(n_total_qubits-i-1)] = h_val
    # local transverse fields
    if np.any(g):
        for i in range(n_total_qubits):
            g_val = g[i] if not np.isscalar(g) else g
            if g_val != 0:
                H['I'*i + 'X' + 'I'*(n_total_qubits-i-1)] = g_val

    if as_dict:
        return H

    # convert the Hamiltonian to a string
    # find all unique coefficients (up to precision)
    coeffs_unique = np.unique(list(H.values()))
    # group terms with the same coefficient
    H_rev = {v: [k for k in H if H[k] == v] for v in coeffs_unique}
    # H_groups = [f'{v}*({' + '.join(H_rev[v])})' if len(H_rev[v]) > 1 else f'{v}*{H_rev[v][0]}' for v in reversed(coeffs_unique)]
    H_groups = []
    for v in reversed(coeffs_unique):
        if len(H_rev[v]) > 1:
            H_group = f' + '.join(H_rev[v])
            if v != 1:
                H_group = f'{v}*({H_group})'
        else:
            H_group = H_rev[v][0]
            if v != 1:
                H_group = f'{v}*{H_group}'
        H_groups.append(H_group)
    H_str = ' + '.join(H_groups)

    # Add I*n - I*n if there are no fields
    if H_str == '':
        H_str += 'I'*n_total_qubits + ' - ' + 'I'*n_total_qubits
    # offset
    if np.any(offset):
        H_str += f' + {offset}'
    return H_str

def ising_graph(graph, J=(-1,1), h=(-1,1), g=(-1,1), offset=0):
    """ Takes a graph and generates a Hamiltonian string for it that is compatible with `ph`. """
    import netket as nk # TODO: add networkx, too

    if not isinstance(graph, nk.graph.Graph):
        raise ValueError(f"graph must be a nk.graph.Graph, but is {type(graph)}")

    # get the number of qubits
    n_qubits = graph.n_nodes
    # get the edges
    edges = graph.edges()
    # get the coupling matrix
    J = np.array(J)
    if J.shape == ():
        # triangular matrix with all couplings set to J
        J = np.triu(np.ones((n_qubits, n_qubits)), k=1) * J
    elif J.shape == (2,):
        # triangular matrix with all couplings set to a random value in this range
        J = np.triu(np.random.uniform(J[0], J[1], (n_qubits, n_qubits)), k=1)
    elif J.shape == (n_qubits, n_qubits):
        # use the given matrix
        pass
    else:
        raise ValueError(f"J must be a scalar, 2-element vector, or matrix of shape {(n_qubits, n_qubits)}, but is {J.shape}")

    # get the longitudinal fields
    if h is not None:
        h = np.array(h)
        if h.shape == ():
            h = np.ones(n_qubits) * h
        elif h.shape == (2,):
            h = np.random.uniform(h[0], h[1], n_qubits)
        elif h.shape == (n_qubits,):
            pass
        else:
            raise ValueError(f"h must be a scalar, 2-element vector, or vector of shape {(n_qubits,)}, but is {h.shape}")

    # get the transverse fields
    if g is not None:
        g = np.array(g)
        if g.shape == ():
            g = np.ones(n_qubits) * g
        elif g.shape == (2,):
            g = np.random.uniform(g[0], g[1], n_qubits)
        elif g.shape == (n_qubits,):
            pass
        else:
            raise ValueError(f"g must be a scalar, 2-element vector, or vector of shape {(n_qubits,)}, but is {g.shape}")

    # generate the Hamiltonian
    H_str = ''
    # pairwise interactions
    for i, j in edges:
        if J[i,j] != 0:
            H_str += str(J[i,j]) + '*' + 'I'*i + 'Z' + 'I'*(j-i-1) + 'Z' + 'I'*(n_qubits-j-1) + ' + '
    # local longitudinal fields
    if np.any(h):
        H_str += ' + '.join([str(h[i]) + '*' + 'I'*i + 'Z' + 'I'*(n_qubits-i-1) for i in range(n_qubits) if h[i] != 0]) + ' + '
    # local transverse fields
    if np.any(g):
        H_str += ' + '.join([str(g[i]) + '*' + 'I'*i + 'X' + 'I'*(n_qubits-i-1) for i in range(n_qubits) if g[i] != 0])

    # offset
    if offset != 0:
        H_str += f" + {offset}"

    return H_str

def get_H_energies(H, expi=False, k=None):
    """Returns the energies of the given hamiltonian `H`. For `expi=True` it gives the same result as `get_pe_energies(exp_i(H))` (up to sorting) and for `expi=False` (default) it returns the eigenvalues of `H`."""
    if type(H) == str:
        H = parse_hamiltonian(H, sparse=k is not None)
    if isinstance(H, np.ndarray):
        energies = np.linalg.eigvalsh(H)
        if k is not None:
            energies = energies[...,:k]  # allow batching for dense matrices
    else:
        energies = sp.linalg.eigsh(H, k=k, which='SA', return_eigenvectors=False)[::-1]  # smallest eigenvalues first
    if expi:
        energies = (energies % (2*np.pi))/(2*np.pi)
        energies[energies > 0.5] -= 1
        # energies = np.sort(energies, axis=-1)
    return energies

def pauli_basis(n, kind='np', normalize=False):
    """ Generate the pauli basis of hermitian 2**n x 2**n matrices. This basis is orthonormal and, except for the identity, traceless. They are also unitary and therefore involutory.

    E.g. for n = 2, the basis is [II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ]

    Parameters
        n (int): Number of qubits
        kind (str): 'np' for numpy arrays (default), 'sp' for scipy sparse matrices, or 'str' for strings
        normalize (bool): Whether to normalize the basis elements (default False)

    Returns
        list[ np.ndarray | scipy.sparse.csr_matrix | str ]: The pauli basis
    """
    def reduce_norm(f, l, normalize):
        if normalize:
            # apply norm np.sqrt(2**n) to the first element, and reduce the rest
            first = l[0]/np.sqrt(2**n)
            if len(l) == 1:
                return first
            rest = reduce(f, l[1:])
            return f(first, rest)
        else:
            return reduce(f, l)

    if kind == 'np':
        return [reduce_norm(np.kron, i, normalize) for i in itertools.product([I,X,Y,Z], repeat=n)]
    elif kind == 'sp':
        basis = [sp.csr_array(b) for b in [I,X,Y,Z]]
        return [reduce_norm(sp.kron, i, normalize) for i in itertools.product(basis, repeat=n)]
    elif kind == 'str':
        norm_str = f"{1/np.sqrt(2**n)}*" if normalize else ""
        return [norm_str + ''.join(i) for i in itertools.product(['I', 'X', 'Y', 'Z'], repeat=n)]
    else:
        raise ValueError(f"Unknown kind: {kind}")

# from https://docs.pennylane.ai/en/stable/code/api/pennylane.pauli_decompose.html
def pauli_decompose(H, eps=1e-5):
    r"""Decomposes a Hermitian matrix into a linear combination of Pauli operators.

    Parameters
        H (ndarray): Hermitian matrix of shape ``(2**n, 2**n)``
        eps (float): Threshold to include a term in the decomposition. Set to 0 to include all terms.

    Returns
        tuple[list[float], list[str]]: the coefficients and the Pauli operator strings

    Example
    >>> H = np.array([[-2, -2+1j, -2, -2], [-2-1j,  0,  0, -1], [-2,  0, -2, -1], [-2, -1, -1,  0]])
    >>> pauli_decompose(H)
    ([-1.0, -1.5, -0.5, -1.0, -1.5, -1.0, -0.5, 1.0, -0.5, -0.5],
     ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XZ', 'YY', 'ZX', 'ZY'])
    """
    n = count_qubits(H)
    N = 2**n

    if H.shape != (N, N):
        raise ValueError(f"The matrix should have shape (2**n, 2**n), for a number of qubits n>=1, but is {H.shape}")

    assert is_hermitian(H), f"The matrix is not Hermitian:\n{H}"

    obs_lst = []
    coeffs = []

    for term, basis_matrix in zip(pauli_basis(n, kind='str'), pauli_basis(n, kind='np')):
        coeff = np.trace(basis_matrix @ H) / N  # project H onto the basis matrix
        coeff = np.real_if_close(coeff).item()

        if abs(coeff) >= eps:
            coeffs.append(coeff)
            obs_lst.append(term)

    return coeffs, obs_lst

###############
### Aliases ###
###############

ph = parse_hamiltonian
pu = parse_unitary
QC = QuantumComputer

#############
### Tests ###
#############

def test_quantum_all():
    tests = [
        _test_constants,
        _test_fourier_matrix,
        _test_parse_unitary,
        _test_parse_hamiltonian,  # required by _test_random_ham
        _test_random_ham,  # required by _test_exp_i
        _test_exp_i,
        _test_get_H_energies_eq_get_pe_energies,
        _test_QuantumComputer,
        _test_random_ket,  # required _test_reverse_qubit_order
        _test_random_dm,   # required by _test_partial_trace
        _test_reverse_qubit_order,
        _test_partial_trace,
        _test_ket_unket,
        _test_op_dm,
        _test_is_dm,
        _test_is_eigenstate,
        _test_count_qubits,
        _test_entropy_von_Neumann,
        _test_entropy_entanglement,
        _test_fidelity,
        _test_trace_distance,
        _test_Schmidt_decomposition,
        _test_correlation_quantum,
        _test_ground_state,
        _test_ising,
        _test_pauli_basis,
        _test_pauli_decompose
    ]

    for test in tests:
        print("Running", test.__name__, "... ", end="", flush=True)
        test()
        print("Test succeeded!", flush=True)

def _test_constants():
    global I, X, Y, Z, H_gate, S, T_gate, CNOT, SWAP
    H = H_gate

    assert is_involutory(X)
    assert is_involutory(Y)
    assert is_involutory(Z)
    assert is_involutory(H)
    assert anticommute(X,Y)
    assert anticommute(Y,Z)
    assert anticommute(X+Y-Z,H)
    assert np.allclose(S @ S, Z)
    assert np.allclose(T_gate @ T_gate, S)
    assert is_involutory(CNOT)
    assert is_involutory(SWAP)
    assert np.allclose(Rx(2*np.pi), -I)
    assert np.allclose(Ry(2*np.pi), -I)
    assert np.allclose(Rz(2*np.pi), -I)

def _test_fourier_matrix():
    assert np.allclose(Fourier_matrix(1), H)
    n = randint(2,8)
    F = Fourier_matrix(n)
    assert is_unitary(F)
    assert np.allclose(F @ ket('0'*n), normalize(np.ones(2**n)))  # Fourier creates a full superposition or ...
    assert np.allclose(F[:,0], parse_unitary('H'*n)[:,0])  # ... in other words

def _test_parse_unitary():
    assert np.allclose(parse_unitary('I'), I)
    assert np.allclose(parse_unitary('X'), X)
    assert np.allclose(parse_unitary('Y'), Y)
    assert np.allclose(parse_unitary('Z'), Z)
    assert np.allclose(parse_unitary('T'), T_gate)
    assert np.allclose(parse_unitary('t'), T_gate.T.conj())
    assert np.allclose(parse_unitary('S'), S)
    assert np.allclose(parse_unitary('s'), S.T.conj())

    assert np.allclose(parse_unitary('CX'), CX)
    assert np.allclose(parse_unitary('XC'), reverse_qubit_order(CX))
    assert np.allclose(parse_unitary('CCX'), Toffoli)
    assert np.allclose(parse_unitary('XCC'), reverse_qubit_order(Toffoli))
    assert np.allclose(parse_unitary('CX @ XC @ CX'), SWAP)
    assert np.allclose(parse_unitary('SS @ HI @ CX @ XC @ IH'), iSWAP)
    assert np.allclose(parse_unitary('IIII @ IIII'), I_(4))
    assert np.allclose(transpose_qubit_order(pu('IXZ'), [0,2,1]), pu('IZX'))

    assert is_unitary(parse_unitary('XCX'))
    assert is_unitary(parse_unitary('CXC'))
    assert is_unitary(parse_unitary('XCXC'))
    assert is_unitary(parse_unitary('XCXCX @ CZCZC'))

    assert np.sum(np.where((parse_unitary('XXX') - I_(3)) != 0)) == 112
    assert np.sum(np.where((parse_unitary('CXX') - I_(3)) != 0)) == 88
    assert np.sum(np.where((parse_unitary('XCX') - I_(3)) != 0)) == 72
    assert np.sum(np.where((parse_unitary('XXC') - I_(3)) != 0)) == 64
    assert np.sum(np.where((parse_unitary('CCX') - I_(3)) != 0)) == 52
    assert np.sum(np.where((parse_unitary('CXC') - I_(3)) != 0)) == 48
    assert np.sum(np.where((parse_unitary('XCC') - I_(3)) != 0)) == 40
    assert np.sum(np.where((parse_unitary('CCC') - I_(3)) != 0)) == 0
    pass

def _test_parse_hamiltonian():
    H = parse_hamiltonian('0.5*(II + ZI - ZX + IX)')
    assert np.allclose(H, CNOT)

    H = parse_hamiltonian('0.5*(XX + YY + ZZ + II)')
    assert np.allclose(H, SWAP)

    H = parse_hamiltonian('-(XX + YY + .5*ZZ) + 1.5')
    assert np.allclose(np.sum(H), 2)

    H = parse_hamiltonian('0.2*(-0.5*(3*XX + 4*YY) + 1*II)')
    assert np.allclose(np.sum(H), -.4)

    H = parse_hamiltonian('X + 2*(I+Z)')
    assert np.allclose(H, X + 2*(I+Z))

    H = parse_hamiltonian('1*(ZZI + IZZ) + 1*(ZII + IZI + IIZ)')
    assert np.allclose(np.sum(np.abs(H)), 14)

    H = parse_hamiltonian('-.25*(ZZI + IZZ) + 1.5')
    assert np.allclose(np.sum(np.abs(H)), 12)

    H = parse_hamiltonian('1.2*IZZI')
    IZZI = np.kron(np.kron(I, Z), np.kron(Z, I))
    assert np.allclose(H, 1.2*IZZI)

def _test_random_ham():
    for _ in range(100):
        n_qubits = np.random.randint(1, 5)
        n_terms = np.random.randint(1, 100)
        n_terms = min(n_terms, 2**(2*n_qubits)-1)
        H = random_ham(n_qubits, n_terms)
        H = parse_hamiltonian(H)
        assert H.shape == (2**n_qubits, 2**n_qubits)
        assert np.allclose(np.trace(H), 0)
        assert is_hermitian(H)

def _test_exp_i():
    n = randint(1,6)
    n_terms = randint(1, 2**(n+1))
    H_str = random_ham(n, n_terms)
    H = ph(H_str)
    U_expect = matexp(1j*H)
    U_actual = get_unitary(exp_i(H))
    assert np.allclose(U_expect, U_actual), f"H_str: {H_str}"
    U_actual = exp_i(H).get_unitary()
    assert np.allclose(U_expect, U_actual), f"H_str: {H_str}"

    U_expect = matexp(1j*3*H)
    U_actual = get_unitary(exp_i(H)**3)
    assert np.allclose(U_expect, U_actual), f"H_str: {H_str}"
    U_actual = (exp_i(H)**3).get_unitary()
    assert np.allclose(U_expect, U_actual), f"H_str: {H_str}"
    U_actual = exp_i(H).get_unitary(3)
    assert np.allclose(U_expect, U_actual), f"H_str: {H_str}"
    U_actual = exp_i(H, k=3).get_unitary()
    assert np.allclose(U_expect, U_actual), f"H_str: {H_str}"

def _test_get_H_energies_eq_get_pe_energies():
    n_qubits = np.random.randint(1, 5)
    n_terms = np.random.randint(1, 100)
    n_terms = min(n_terms, 2**(2*n_qubits)-1)
    H = random_ham(n_qubits, n_terms, scaling=False)
    H = parse_hamiltonian(H)

    A = np.sort(get_pe_energies(exp_i(H)))
    B = np.sort(get_H_energies(H, expi=True))
    assert np.allclose(A, B), f"{A} ≠ {B}"

def _test_ket_unket():
    assert np.allclose(ket('0'), [1,0])
    assert np.allclose(ket('1'), [0,1])
    assert np.allclose(ket(0), [1,0])
    assert np.allclose(ket(2), [0,0,1,0])
    assert np.allclose(ket('00 + 11 + 01 + 10'), [.5,.5,.5,.5])
    assert np.allclose(ket('00+ 11- 01 -10'), [.5,-.5,-.5,.5])
    assert np.allclose(ket('00 + 11'), ket('2*00 + 2*11'))
    # assert np.allclose(ket('00 - 11'), ket('2*(00 - 11)'))  # parentheses NYI
    assert np.allclose(ket('3*00 + 4*01 + 5*11'), [sqrt(9/50), sqrt(16/50), 0, sqrt(25/50)])
    assert unket(ket('1010')) == '1010'
    assert unket(ket(10)) == '1010'
    assert unket(ket(10, 5)) == '01010'
    # ket should be fast enough to return already-kets 1000 times in negligible time
    import time
    psi = random_ket(2)
    max_time = 0.02
    start = time.time()
    for _ in range(10):
        assert time.time() - start < max_time, f"ket is too slow (iteration {_}/5)"
        for _ in range(100):
            ket(psi)

def _test_random_ket():
    for _ in range(10):
        n_qubits = np.random.randint(1, 10)
        psi = random_ket(n_qubits)
        assert psi.shape == (2**n_qubits,)
        assert np.allclose(np.linalg.norm(psi), 1)

def _test_op_dm():
    assert np.allclose(op(0),   [[1,0], [0,0]])
    assert np.allclose(op(0,1), [[0,1], [0,0]])
    assert np.allclose(op(1,0), [[0,0], [1,0]])
    assert np.allclose(op(1),   [[0,0], [0,1]])
    O = op(0,3,n=2)
    assert O.shape[0] == O.shape[1]
    # op should be fast enough to return already-density-matrices 1000 times in negligible time
    import time
    rho = random_dm(2)
    max_time = 0.01
    start = time.time()
    for _ in range(10):
        assert time.time() - start < max_time, f"op is too slow (iteration {_}/10)"
        for _ in range(100):
            op(rho)

def _test_is_dm():
    assert is_dm(np.eye(2**2)/2**2)
    # random Bloch vectors
    for _ in range(10):
        v = np.random.uniform(-1, 1, 3)
        if np.linalg.norm(v) > 1:
            v = normalize(v)
        # create dm from Bloch vector
        rho = (I + v[0]*X + v[1]*Y + v[2]*Z)/2
        assert is_dm(rho)

def _test_random_dm():
    for _ in range(100):
        n_qubits = np.random.randint(1, 5)
        rho = random_dm(n_qubits)
        assert is_dm(rho)

def _test_QuantumComputer():
    # test basic functionality
    qc = QuantumComputer()
    qc.x(0)
    assert unket(qc.get_state()) == '1'
    qc.cx(0, 1)
    assert unket(qc.get_state()) == '11'
    qc.reset()
    assert unket(qc.get_state()) == '00'
    qc.h()
    qc.x(2)
    qc.init(2, [0,1])
    assert unket(qc.get_state()) == '101'
    qc.z([0,2])  # noop
    qc.swap(1,2)
    assert unket(qc.get_state()) == '110'
    qc.reset(1)
    assert unket(qc.get_state()) == '100'
    qc.remove([0,2])
    assert unket(qc.get_state()) == '0'
    qc.x(2)
    result = qc.measure('all')
    assert result == '01'

    # Heisenberg uncertainty principle
    qc = QuantumComputer(1)
    qc.init(random_ket(1))
    assert qc.std(X) * qc.std(Z) >= abs(qc.ev(1j*(X@Z - Z@X)))/2

    # Bell basis
    Bell = [
        ket('00 + 11'),
        ket('00 - 11'),
        ket('01 + 10'),
        ket('01 - 10')
    ]
    obs = sum(i*op(b) for i, b in zip(range(1,5), Bell))
    qc = QuantumComputer(2)
    qc.h(0)
    qc.cx(0, 1)
    p = qc.probs(obs=obs)
    assert np.isclose(entropy(p), 0), f"p = {p}"
    U_expected = pu('CX @ HI')
    assert np.allclose(qc.U, U_expected), f"Incorrect unitary:\n{qc.U}\n ≠\n{U_expected}"

    # more complex test
    qc = QuantumComputer(15)
    U = parse_unitary('XYZCZYX')
    qc(U, choice(qc.qubits, 7, False))
    qc.remove([5,7,6,10,2])
    U = random_unitary(2**5)
    qc(U, choice(qc.qubits, 5, False))
    assert qc.n == 10

    # test phase estimation
    H = ph(f'{1/8}*(IZ + ZI + II)')
    U = exp_i(2*np.pi*H)
    assert np.isclose(np.trace(H @ op('00')), float_from_binstr('.011'))  # 00 is eigenstate with energy 0.375 = '011'
    state_qubits = ['s0', 's1']
    qc = QuantumComputer(state_qubits)
    E_qubits = ['e0', 'e1', 'e2']
    qc.pe(U, state_qubits, E_qubits)
    res = qc.measure(E_qubits)
    assert res == '011', f"measurement result was {res} ≠ '011'"

    qc.remove('s0')
    assert np.allclose(qc.get_state(), ket('0011'))
    qc.remove('all')

def _test_reverse_qubit_order():
    # known 3-qubit matrix
    psi = np.kron(np.kron([1,1], [0,1]), [1,-1])
    psi_rev1 = np.kron(np.kron([1,-1], [0,1]), [1,1])
    psi_rev2 = reverse_qubit_order(psi)
    assert np.allclose(psi_rev1, psi_rev2)

    # same as above, but with n random qubits
    n = 10
    psis = [random_ket(1) for _ in range(n)]
    psi = psis[0]
    for i in range(1,n):
        psi = np.kron(psi, psis[i])
    psi_rev1 = psis[-1]
    for i in range(1,n):
        psi_rev1 = np.kron(psi_rev1, psis[-i-1])

    psi_rev2 = reverse_qubit_order(psi)
    assert np.allclose(psi_rev1, psi_rev2)

    # general hamiltonian
    H = parse_hamiltonian('IIIXX')
    H_rev1 = parse_hamiltonian('XXIII')
    H_rev2 = reverse_qubit_order(H)
    assert np.allclose(H_rev1, H_rev2)

    H = parse_hamiltonian('XI + YI')
    H_rev1 = parse_hamiltonian('IX + IY')
    H_rev2 = reverse_qubit_order(H)
    assert np.allclose(H_rev1, H_rev2), f"{H_rev1} \n≠\n {H_rev2}"

    # pure density matrix
    psi = np.kron(np.kron([1,1], [0,1]), [1,-1])
    rho = np.outer(psi, psi)
    psi_rev1 = np.kron(np.kron([1,-1], [0,1]), [1,1])
    rho_rev1 = np.outer(psi_rev1, psi_rev1)
    rho_rev2 = reverse_qubit_order(rho)
    assert np.allclose(rho_rev1, rho_rev2)

    # draw n times 2 random 1-qubit states and a probability distribution over all n pairs
    n = 10
    psis = [[random_dm(1) for _ in range(2)] for _ in range(n)]
    p = normalize(np.random.rand(n), p=1)
    # compute the average state
    psi = np.zeros((2**2, 2**2), dtype=complex)
    for i in range(n):
        psi += p[i]*np.kron(psis[i][0], psis[i][1])
    # compute the average state with reversed qubit order
    psi_rev1 = np.zeros((2**2, 2**2), dtype=complex)
    for i in range(n):
        psi_rev1 += p[i]*np.kron(psis[i][1], psis[i][0])

    psi_rev2 = reverse_qubit_order(psi)
    assert np.allclose(psi_rev1, psi_rev2), f"psi_rev1 = {psi_rev1}\npsi_rev2 = {psi_rev2}"

def _test_partial_trace():
    # known 4x4 matrix
    rho = np.arange(16).reshape(4,4)
    rhoA_expect = np.array([[ 5, 9], [21, 25]])
    rhoA_actual = partial_trace(rho, 0)
    assert np.allclose(rhoA_expect, rhoA_actual), f"rho_expect = {rhoA_expect}\nrho_actual = {rhoA_actual}"

    # two separable density matrices
    rhoA = random_dm(2)
    rhoB = random_dm(3)
    rho = np.kron(rhoA, rhoB)
    rhoA_expect = rhoA
    rhoA_actual = partial_trace(rho, [0,1])
    assert np.allclose(rhoA_expect, rhoA_actual), f"rho_expect = {rhoA_expect}\nrho_actual = {rhoA_actual}"

    # two separable state vectors
    psiA = random_ket(2)
    psiB = random_ket(3)
    psi = np.kron(psiA, psiB)
    psiA_expect = np.outer(psiA, psiA.conj())
    psiA_actual = partial_trace(psi, [0,1])
    assert np.allclose(psiA_expect, psiA_actual), f"psi_expect = {psiA_expect}\npsi_actual = {psiA_actual}"

    # total trace
    st = random_ket(3)
    st_tr = partial_trace(st, [])
    assert np.allclose(np.array([[1]]), st_tr), f"st_tr = {st_tr} ≠ 1"
    rho = random_dm(3)
    rho_tr = partial_trace(rho, [])
    assert np.allclose(np.array([[1]]), rho_tr), f"rho_expect = {rhoA_expect}\nrho_actual = {rhoA_actual}"

    # retain all qubits
    st = random_ket(3)
    st_tr = partial_trace(st, [0,1,2])
    st_expect = np.outer(st, st.conj())
    assert st_expect.shape == st_tr.shape, f"st_expect.shape = {st_expect.shape} ≠ st_tr.shape = {st_tr.shape}"
    assert np.allclose(st_expect, st_tr), f"st_expect = {st_expect} ≠ st_tr = {st_tr}"
    rho = random_dm(2)
    rho_tr = partial_trace(rho, [0,1])
    assert rho.shape == rho_tr.shape, f"rho.shape = {rho.shape} ≠ rho_tr.shape = {rho_tr.shape}"
    assert np.allclose(rho, rho_tr), f"rho_expect = {rhoA_expect}\nrho_actual = {rhoA_actual}"

def _test_is_eigenstate():
    H = parse_hamiltonian('XX + YY + ZZ')
    assert is_eigenstate(ket('00'), H)
    assert not is_eigenstate(ket('01'), H)

def _test_count_qubits():
    assert count_qubits(ising(20)) == 20
    assert count_qubits('CXC @ XCC') == 3
    assert count_qubits(parse_unitary('CXC @ XCC')) == 3
    assert count_qubits('0.001*01 - 0.101*11') == 2
    assert count_qubits(ket('0.001*01 - 0.101*11')) == 2
    qc = QuantumComputer(4)
    qc.x(4)
    assert count_qubits(qc) == 5

def _test_entropy_von_Neumann():
    rho = random_dm(2, pure=True)
    S = entropy_von_Neumann(rho)
    assert np.allclose(S, 0), f"S = {S} ≠ 0"

    rho = np.eye(2)/2
    S = entropy_von_Neumann(rho)
    assert np.allclose(S, 1), f"S = {S} ≠ 1"

def _test_entropy_entanglement():
    # Two qubits in the Bell state |00> + |11> should have entropy 1
    rho = op('00 + 11')
    S = entropy_entanglement(rho, [0])
    assert np.allclose(S, 1), f"S = {S} ≠ 1"

    # Two separable systems should have entropy 0
    rhoA = random_dm(2, pure=True)
    rhoB = random_dm(3, pure=True)
    rho = np.kron(rhoA, rhoB)
    S = entropy_entanglement(rho, [0,1])
    assert np.allclose(S, 0), f"S = {S} ≠ 0"

def _test_fidelity():
    # same state
    psi = random_ket(2)
    assert np.allclose(fidelity(psi, psi), 1), f"fidelity = {fidelity(psi, psi)} ≠ 1"
    rho = random_dm(2)
    assert np.allclose(fidelity(rho, rho), 1), f"fidelity = {fidelity(rho, rho)} ≠ 1"

    # orthogonal states
    psi1 = ket('00')
    psi2 = ket('11')
    assert np.allclose(fidelity(psi1, psi2), 0), f"fidelity = {fidelity(psi1, psi2)} ≠ 0"
    rho1 = dm('00')
    rho2 = dm('11')
    assert np.allclose(fidelity(rho1, rho2), 0), f"fidelity = {fidelity(rho1, rho2)} ≠ 0"
    assert np.allclose(fidelity(psi1, rho2), 0), f"fidelity = {fidelity(psi1, rho2)} ≠ 0"
    assert np.allclose(fidelity(rho1, psi2), 0), f"fidelity = {fidelity(rho1, psi2)} ≠ 0"

    # check values
    psi1 = ket('0')
    psi2 = ket('-')
    rho1 = dm('0')
    rho2 = dm('-')
    assert np.allclose(fidelity(psi1, psi2), 1/2), f"fidelity = {fidelity(psi1, psi2)} ≠ 1/2"
    assert np.allclose(fidelity(rho1, rho2), 1/2), f"fidelity = {fidelity(rho1, rho2)} ≠ 1/2"
    assert np.allclose(fidelity(psi1, rho2), 1/2), f"fidelity = {fidelity(psi1, rho2)} ≠ 1/2"
    assert np.allclose(fidelity(rho1, psi2), 1/2), f"fidelity = {fidelity(rho1, psi2)} ≠ 1/2"

    # random states to test properties: F(s1,s2) \in [0,1], symmetric
    psi1 = random_ket(2)
    psi2 = random_ket(2)
    assert 0 <= fidelity(psi1, psi2) <= 1, f"fidelity = {fidelity(psi1, psi2)} ∉ [0,1]"
    assert np.allclose(fidelity(psi1, psi2), fidelity(psi2, psi1)), f"fidelity(psi1, psi2) = {fidelity(psi1, psi2)} ≠ {fidelity(psi2, psi1)}"
    rho1 = random_dm(2)
    rho2 = random_dm(2)
    assert 0 <= fidelity(rho1, rho2) <= 1, f"fidelity = {fidelity(rho1, rho2)} ∉ [0,1]"
    assert np.allclose(fidelity(rho1, rho2), fidelity(rho2, rho1)), f"fidelity(rho1, rho2) = {fidelity(rho1, rho2)} ≠ {fidelity(rho2, rho1)}"
    assert np.allclose(fidelity(psi1, rho2), fidelity(rho2, psi1)), f"fidelity(psi1, rho2) = {fidelity(psi1, rho2)} ≠ {fidelity(rho2, psi1)}"
    assert np.allclose(fidelity(rho1, psi2), fidelity(psi2, rho1)), f"fidelity(rho1, psi2) = {fidelity(rho1, psi2)} ≠ {fidelity(psi2, rho1)}"
    assert 0 <= fidelity(psi1, rho2) <= 1, f"fidelity = {fidelity(psi1, rho2)} ∉ [0,1]"
    assert 0 <= fidelity(rho1, psi2) <= 1, f"fidelity = {fidelity(rho1, psi2)} ∉ [0,1]"

def _test_trace_distance():
    # same state
    psi = random_ket(2)
    assert np.isclose(trace_distance(psi, psi), 0), f"trace_distance = {trace_distance(psi, psi)} ≠ 0"
    rho = random_dm(2)
    assert np.isclose(trace_distance(rho, rho), 0), f"trace_distance = {trace_distance(rho, rho)} ≠ 0"
    # orthogonal states
    psi1, psi2 = '00', '11'
    assert np.isclose(trace_distance(psi1, psi2), 1), f"trace_distance = {trace_distance(psi1, psi2)} ≠ 1"
    # other values
    psi1, psi2 = '0', '-'
    assert np.isclose(trace_distance(psi1, psi2), fs(2)), f"trace_distance = {trace_distance(psi1, psi2)} ≠ 1/sqrt(2)"

def _test_Schmidt_decomposition():
    n = 6
    subsystem = np.random.choice(n, size=np.random.randint(1, n), replace=False)
    subsystem = sorted(subsystem)  # TODO: figure out how to correctly transpose axes in partial trace with unsorted subsystems
    psi = random_ket(n)
    l, A, B = Schmidt_decomposition(psi, subsystem)

    # check non-negativity of Schmidt coefficients
    assert np.all(l >= 0), f"l = {l} < 0"
    # check normalization of Schmidt coefficients
    assert np.allclose(np.sum(l**2), 1), f"sum(l**2) = {np.sum(l**2)} ≠ 1"

    # check RDM for subsystem A
    rho_expect = partial_trace(np.outer(psi, psi.conj()), subsystem)
    rho_actual = np.sum([l_i**2 * np.outer(A_i, A_i.conj()) for l_i, A_i in zip(l, A)], axis=0)
    assert np.allclose(rho_expect, rho_actual), f"rho_expect - rho_actual = {rho_expect - rho_actual}"

    # check RDM for subsystem B
    rho_expect = partial_trace(np.outer(psi, psi.conj()), [i for i in range(n) if i not in subsystem])
    rho_actual = np.sum([l_i**2 * np.outer(B_i, B_i.conj()) for l_i, B_i in zip(l, B)], axis=0)
    assert np.allclose(rho_expect, rho_actual), f"rho_expect - rho_actual = {rho_expect - rho_actual}"

    # check entanglement entropy
    S_expect = entropy_entanglement(psi, subsystem)
    S_actual = -np.sum([l_i**2 * np.log2(l_i**2) for l_i in l])
    assert np.allclose(S_expect, S_actual), f"S_expect = {S_expect} ≠ S_actual = {S_actual}"

def _test_correlation_quantum():
    assert np.isclose(correlation_quantum(ket('0101 + 0000'), ZZ, ZZ), 1)
    assert np.isclose(correlation_quantum(ket('0101 + 0000'), XX, XX), 0)
    assert np.isclose(correlation_quantum(ket('0101 + 1010'), XX, XX), 1)
    assert np.isclose(correlation_quantum(ket('0101 + 1010'), ZZ, ZZ), 0)
    assert np.isclose(correlation_quantum(ket('0.5*0101 + 0000'), ZZ, ZZ), 0.64)
    assert np.isclose(correlation_quantum(dm('0.5*0101 + 0000'), ZZ, ZZ), 0.64)

def _test_ground_state():
    H = parse_hamiltonian('ZZII + IZZI + IIZZ', dtype=float)
    ge, gs = -3, ket('0101')

    res_exact = ground_state_exact(H)
    assert np.allclose(res_exact[0], ge)
    assert np.allclose(res_exact[1], gs)

    res_ITE = ground_state_ITE(H)
    assert np.allclose(res_ITE[0], ge)
    # assert np.allclose(res_ITE[1], gs) # might be complex due to random state initialization


def _test_ising():
    # 1d
    H_str = ising(5, J=1.5, h=0, g=0, offset=0, kind='1d', circular=False)
    expect = "1.5*(ZZIII + IZZII + IIZZI + IIIZZ)"
    assert np.allclose(ph(H_str), ph(expect)), f"\nH_str    = {H_str}\nexpect = {expect}"

    H_str = ising(3, J=0, h=3, g=2)
    expect = '3*(ZII + IZI + IIZ) + 2*(XII + IXI + IIX)'
    assert np.allclose(ph(H_str), ph(expect)), f"\nH_str  = {H_str}\nexpect = {expect}"

    H_str = ising(5, J=1.5, h=1.1, g=0.5, offset=0.5, kind='1d', circular=True)
    expect = "1.5*(ZZIII + IZZII + IIZZI + IIIZZ + ZIIIZ) + 1.1*(ZIIII + IZIII + IIZII + IIIZI + IIIIZ) + 0.5*(XIIII + IXIII + IIXII + IIIXI + IIIIX) + 0.5"
    assert np.allclose(ph(H_str), ph(expect)), f"\nH_str  = {H_str}\nexpect = {expect}"

    H_str = ising(3, J=[0.6,0.7,0.8], h=[0.1,0.2,0.7], g=[0.6,0.1,1.5], offset=0.5, kind='1d', circular=True)
    expect = "0.6*ZZI + 0.7*IZZ + 0.8*ZIZ + 0.1*ZII + 0.2*IZI + 0.7*IIZ + 0.6*XII + 0.1*IXI + 1.5*IIX + 0.5"
    assert np.allclose(ph(H_str), ph(expect)), f"\nH_str  = {H_str}\nexpect = {expect}"

    H_str = ising(3, J=[0,1], h=[1,2], g=[2,5], offset=0.5, kind='1d', circular=True)
    # random, but count terms in H_str instead
    n_terms = len(H_str.split('+'))
    assert n_terms == 10, f"n_terms = {n_terms}\nexpect = 10"

    # 2d
    H_str = ising((2,2), J=1.5, h=0, g=0, offset=0, kind='2d', circular=False)
    expect = "1.5*(ZIZI + ZZII + IZIZ + IIZZ)"
    assert H_str == expect, f"\nH_str  = {H_str}\nexpect = {expect}"

    H_str = ising((3,3), J=1.5, h=1.1, g=0.5, offset=0.5, kind='2d', circular=True)
    expect = "1.5*(ZIIZIIIII + ZZIIIIIII + IZIIZIIII + IZZIIIIII + IIZIIZIII + ZIZIIIIII + IIIZIIZII + IIIZZIIII + IIIIZIIZI + IIIIZZIII + IIIIIZIIZ + IIIZIZIII + IIIIIIZZI + ZIIIIIZII + IIIIIIIZZ + IZIIIIIZI + IIZIIIIIZ + IIIIIIZIZ) + 1.1*(ZIIIIIIII + IZIIIIIII + IIZIIIIII + IIIZIIIII + IIIIZIIII + IIIIIZIII + IIIIIIZII + IIIIIIIZI + IIIIIIIIZ) + 0.5*(XIIIIIIII + IXIIIIIII + IIXIIIIII + IIIXIIIII + IIIIXIIII + IIIIIXIII + IIIIIIXII + IIIIIIIXI + IIIIIIIIX) + 0.5"
    assert np.allclose(ph(H_str, sparse=True), ph(expect, sparse=True)), f"\nH_str  = {H_str}\nexpect = {expect}"

    # 3d
    H_str = ising((2,2,3), kind='3d', J=1.2, h=0, g=0, offset=0, circular=True)
    expect = "1.2*(ZIIIIIZIIIII + ZIIZIIIIIIII + ZZIIIIIIIIII + IZIIIIIZIIII + IZIIZIIIIIII + IZZIIIIIIIII + IIZIIIIIZIII + IIZIIZIIIIII + ZIZIIIIIIIII + IIIZIIIIIZII + IIIZZIIIIIII + IIIIZIIIIIZI + IIIIZZIIIIII + IIIIIZIIIIIZ + IIIZIZIIIIII + IIIIIIZIIZII + IIIIIIZZIIII + IIIIIIIZIIZI + IIIIIIIZZIII + IIIIIIIIZIIZ + IIIIIIZIZIII + IIIIIIIIIZZI + IIIIIIIIIIZZ + IIIIIIIIIZIZ)"
    assert H_str == expect, f"\nH_str  = {H_str}\nexpect = {expect}"

    H_str = ising((3,3,3), kind='3d', J=1.5, h=0, g=0, offset=0, circular=True)
    expect = "1.5*(ZIIIIIIIIZIIIIIIIIIIIIIIIII + ZIIZIIIIIIIIIIIIIIIIIIIIIII + ZZIIIIIIIIIIIIIIIIIIIIIIIII + IZIIIIIIIIZIIIIIIIIIIIIIIII + IZIIZIIIIIIIIIIIIIIIIIIIIII + IZZIIIIIIIIIIIIIIIIIIIIIIII + IIZIIIIIIIIZIIIIIIIIIIIIIII + IIZIIZIIIIIIIIIIIIIIIIIIIII + ZIZIIIIIIIIIIIIIIIIIIIIIIII + IIIZIIIIIIIIZIIIIIIIIIIIIII + IIIZIIZIIIIIIIIIIIIIIIIIIII + IIIZZIIIIIIIIIIIIIIIIIIIIII + IIIIZIIIIIIIIZIIIIIIIIIIIII + IIIIZIIZIIIIIIIIIIIIIIIIIII + IIIIZZIIIIIIIIIIIIIIIIIIIII + IIIIIZIIIIIIIIZIIIIIIIIIIII + IIIIIZIIZIIIIIIIIIIIIIIIIII + IIIZIZIIIIIIIIIIIIIIIIIIIII + IIIIIIZIIIIIIIIZIIIIIIIIIII + IIIIIIZZIIIIIIIIIIIIIIIIIII + ZIIIIIZIIIIIIIIIIIIIIIIIIII + IIIIIIIZIIIIIIIIZIIIIIIIIII + IIIIIIIZZIIIIIIIIIIIIIIIIII + IZIIIIIZIIIIIIIIIIIIIIIIIII + IIIIIIIIZIIIIIIIIZIIIIIIIII + IIZIIIIIZIIIIIIIIIIIIIIIIII + IIIIIIZIZIIIIIIIIIIIIIIIIII + IIIIIIIIIZIIIIIIIIZIIIIIIII + IIIIIIIIIZIIZIIIIIIIIIIIIII + IIIIIIIIIZZIIIIIIIIIIIIIIII + IIIIIIIIIIZIIIIIIIIZIIIIIII + IIIIIIIIIIZIIZIIIIIIIIIIIII + IIIIIIIIIIZZIIIIIIIIIIIIIII + IIIIIIIIIIIZIIIIIIIIZIIIIII + IIIIIIIIIIIZIIZIIIIIIIIIIII + IIIIIIIIIZIZIIIIIIIIIIIIIII + IIIIIIIIIIIIZIIIIIIIIZIIIII + IIIIIIIIIIIIZIIZIIIIIIIIIII + IIIIIIIIIIIIZZIIIIIIIIIIIII + IIIIIIIIIIIIIZIIIIIIIIZIIII + IIIIIIIIIIIIIZIIZIIIIIIIIII + IIIIIIIIIIIIIZZIIIIIIIIIIII + IIIIIIIIIIIIIIZIIIIIIIIZIII + IIIIIIIIIIIIIIZIIZIIIIIIIII + IIIIIIIIIIIIZIZIIIIIIIIIIII + IIIIIIIIIIIIIIIZIIIIIIIIZII + IIIIIIIIIIIIIIIZZIIIIIIIIII + IIIIIIIIIZIIIIIZIIIIIIIIIII + IIIIIIIIIIIIIIIIZIIIIIIIIZI + IIIIIIIIIIIIIIIIZZIIIIIIIII + IIIIIIIIIIZIIIIIZIIIIIIIIII + IIIIIIIIIIIIIIIIIZIIIIIIIIZ + IIIIIIIIIIIZIIIIIZIIIIIIIII + IIIIIIIIIIIIIIIZIZIIIIIIIII + IIIIIIIIIIIIIIIIIIZIIZIIIII + IIIIIIIIIIIIIIIIIIZZIIIIIII + ZIIIIIIIIIIIIIIIIIZIIIIIIII + IIIIIIIIIIIIIIIIIIIZIIZIIII + IIIIIIIIIIIIIIIIIIIZZIIIIII + IZIIIIIIIIIIIIIIIIIZIIIIIII + IIIIIIIIIIIIIIIIIIIIZIIZIII + IIZIIIIIIIIIIIIIIIIIZIIIIII + IIIIIIIIIIIIIIIIIIZIZIIIIII + IIIIIIIIIIIIIIIIIIIIIZIIZII + IIIIIIIIIIIIIIIIIIIIIZZIIII + IIIZIIIIIIIIIIIIIIIIIZIIIII + IIIIIIIIIIIIIIIIIIIIIIZIIZI + IIIIIIIIIIIIIIIIIIIIIIZZIII + IIIIZIIIIIIIIIIIIIIIIIZIIII + IIIIIIIIIIIIIIIIIIIIIIIZIIZ + IIIIIZIIIIIIIIIIIIIIIIIZIII + IIIIIIIIIIIIIIIIIIIIIZIZIII + IIIIIIIIIIIIIIIIIIIIIIIIZZI + IIIIIIZIIIIIIIIIIIIIIIIIZII + IIIIIIIIIIIIIIIIIIZIIIIIZII + IIIIIIIIIIIIIIIIIIIIIIIIIZZ + IIIIIIIZIIIIIIIIIIIIIIIIIZI + IIIIIIIIIIIIIIIIIIIZIIIIIZI + IIIIIIIIZIIIIIIIIIIIIIIIIIZ + IIIIIIIIIIIIIIIIIIIIZIIIIIZ + IIIIIIIIIIIIIIIIIIIIIIIIZIZ)"
    assert H_str == expect, f"\nH_str  = {H_str}\nexpect = {expect}"

    # pairwise
    H_str = ising(4, J=-.5, h=.4, g=.7, offset=1, kind='pairwise')
    expect = "-0.5*(ZZII + ZIZI + ZIIZ + IZZI + IZIZ + IIZZ) + 0.4*(ZIII + IZII + IIZI + IIIZ) + 0.7*(XIII + IXII + IIXI + IIIX) + 1"
    assert np.allclose(ph(H_str), ph(expect)), f"\nH_str  = {H_str}\nexpect = {expect}"

    # full
    H_str = ising(3, J=1.5, h=.4, g=.7, offset=1, kind='all')
    expect = "1.5*(ZZI + ZIZ + IZZ + ZZZ) + 0.4*(ZII + IZI + IIZ) + 0.7*(XII + IXI + IIX) + 1"
    assert np.allclose(ph(H_str), ph(expect)), f"\nH_str  = {H_str}\nexpect = {expect}"

    H_str = ising(3, kind='all', J={(0,1): 2, (0,1,2): 3, (1,2):0}, g=0, h=1.35)
    expect = "2*ZZI + 3*ZZZ + 1.35*(ZII + IZI + IIZ)"
    assert np.allclose(ph(H_str), ph(expect)), f"\nH_str  = {H_str}\nexpect = {expect}"

    J_dict = {
        (0,1): 1.5,
        (0,2): 2,
        (1,2): 0.5,
        (0,1,2): 3,
        (0,1,2,3): 0.5
    }
    H_str = ising(4, J=J_dict, h=.3, g=.5, offset=1.2, kind='all')
    expect = "1.5*ZZII + 2*ZIZI + 0.5*IZZI + 3*ZZZI + 0.5*ZZZZ + 0.3*(ZIII + IZII + IIZI + IIIZ) + 0.5*(XIII + IXII + IIXI + IIIX) + 1.2"
    assert np.allclose(ph(H_str), ph(expect)), f"\nH_str  = {H_str}\nexpect = {expect}"

def _test_pauli_basis():
    n = np.random.randint(1,4)
    pauli_n = pauli_basis(n)

    # check the number of generators
    n_expect = 2**(2*n)
    assert len(pauli_n) == n_expect, f"Number of generators is {len(pauli_n)}, but should be {n_expect}!"

    # check if no two generators are the same
    for i, (A,B) in enumerate(itertools.combinations(pauli_n,2)):
        assert not np.allclose(A, B), f"Pair {i} is not different!"

    # check if all generators except of the identity are traceless
    assert np.allclose(pauli_n[0], np.eye(2**n)), "First generator is not the identity!"
    for i, A in enumerate(pauli_n[1:]):
        assert np.isclose(np.trace(A), 0), f"Generator {i} is not traceless!"

    # check if all generators are Hermitian
    for i, A in enumerate(pauli_n):
        assert is_hermitian(A), f"Generator {i} is not Hermitian!"

    # check if all generators are orthogonal
    for i, (A,B) in enumerate(itertools.combinations(pauli_n,2)):
        assert np.allclose(np.trace(A.conj().T @ B), 0), f"Pair {i} is not orthogonal!"

    # check normalization
    pauli_n_norm = pauli_basis(n, kind='np', normalize=True)
    for i, A in enumerate(pauli_n_norm):
        assert np.isclose(np.linalg.norm(A), 1), f"Generator {i} does not have norm 1!"

    # check string representation
    pauli_n_str = pauli_basis(n, kind='str')
    assert len(pauli_n) == len(pauli_n_str), "Number of generators is not the same!"

    # check if all generators are the same
    for i, (A,B) in enumerate(zip(pauli_n, pauli_n_str)):
        assert np.allclose(A, parse_hamiltonian(B)), f"Generator {i} is not the same!"

    # check sparse representation
    pauli_n_sp = pauli_basis(n, kind='sp')
    assert len(pauli_n) == len(pauli_n_sp), "Number of generators is not the same!"

    # check if all generators are the same
    for i, (A,B) in enumerate(zip(pauli_n, pauli_n_sp)):
        assert np.allclose(A, B.todense()), f"Generator {i} is not the same!"

def _test_pauli_decompose():
    global f2, H_gate, SWAP
    H = H_gate

    # H = (X+Z)/sqrt(2)
    coeff, basis = pauli_decompose(H)
    assert np.allclose(coeff, [f2]*2), f"coeff = {coeff} ≠ [{f2}]*2"
    assert basis == ['X', 'Z'], f"basis = {basis} ≠ ['X', 'Z']"

    # SWAP = 0.5*(II + XX + YY + ZZ)
    coeff, basis = pauli_decompose(SWAP)
    assert np.allclose(coeff, [0.5]*4), f"coeff = {coeff} ≠ [0.5]*4"
    assert basis == ['II', 'XX', 'YY', 'ZZ'], f"basis = {basis} ≠ ['II', 'XX', 'YY', 'ZZ']"

    # random 2-qubit hamiltonian
    H = random_hermitian(4)
    coeff, basis = pauli_decompose(H)
    assert len(coeff) == 16, f"len(coeff) = {len(coeff)} ≠ 16"
    assert len(basis) == 16, f"len(basis) = {len(basis)} ≠ 16"

    # check if the decomposition is correct
    H_decomposed = np.zeros((4,4), dtype=complex)
    for c, b in zip(coeff, basis):
        H_decomposed += c*parse_hamiltonian(b)
    assert np.allclose(H, H_decomposed), f"H = {H}\nH_decomposed = {H_decomposed}"

    # check if `include_zero` returns the whole basis
    n = 4
    coeff, basis = pauli_decompose(np.eye(2**n), eps=0)
    n_expect = 2**(2*n)  # == len(pauli_basis(n))
    assert len(coeff) == n_expect, f"len(coeff) = {len(coeff)} ≠ {n_expect}"
    assert len(basis) == n_expect, f"len(basis) = {len(basis)} ≠ {n_expect}"
