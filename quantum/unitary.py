import sys, warnings
import numpy as np
from functools import reduce
try:
    from scipy.linalg import eig, eigh
except ImportError:
    from numpy.linalg import eig, eigh
from math import log2, sqrt

from .constants import I_, I, X, Y, Z, S, T_gate, H  # used in parse_unitary -> globals()
from .state import ket, op, count_qubits, plotQ, reverse_qubit_order, transpose_qubit_order
from ..mathlib import is_unitary, is_hermitian, pauli_decompose, count_bitreversed
from ..utils import is_int

def Fourier_matrix(n, swap=False):
    """ Calculate the Fourier matrix of size `n`. The Fourier matrix is the matrix representation of the quantum Fourier transform (QFT).
    If swap == False, the matrix is $F_{jk} = \\frac{1}{\\sqrt{n}} e^{2\\pi i jk/n}$.
    If swap == True, the matrix is $F_{jk} = \\frac{1}{\\sqrt{n}} e^{2\\pi i \\overline{j} k/n}$, where $\\overline{j}$ is `j` bit-reverse.
    """
    if swap:
        q = int(log2(n))
        assert n == 2**q, f'Only can swap if n is a power of two, but was: {n} ≠ 2**{q}'
        row_order = count_bitreversed(q)
    else:
        row_order = range(n)
    cols, rows = np.meshgrid(range(n), row_order)
    return np.exp(2j*np.pi/n * cols * rows)/sqrt(n)
    # return np.array([[np.exp(2j*np.pi*j*k/n) for k in range(n)] for j in row_order])/sqrt(n)

def parse_unitary(unitary, check=2):
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

    if check >= 2:
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
                warnings.warn("shots=0 is not supported for statevector_simulator. Using shots=1 instead.", stacklevel=2)
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
            - `use_pauli == False` calculates full diagonalization and stores two dxd matrices (+ one dx1 vector).
            - `use_pauli == True` uses `PauliEvolutionGate`, which has lower memory footprint if the Hamiltonian can be expressed
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
                        warnings.warn(f"Diagonalizing a {self.n}-qubit matrix", stacklevel=2)
                    assert is_hermitian(H), "Hamiltonian must be hermitian"
                    self.D, self.U = eigh(H)
            # auxiliary variables
            self.k = int(k) if is_int(k) else k
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
                return exp_i(self.pauli_ev, k=self.k*k, use_pauli=True)
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
            return self.U @ (np.exp(self.k*k*1j*self.D)[:,None] * self.U.T.conj())

    def get_unitary(circ, decimals=None, as_np=True):
        if hasattr(circ, 'get_unitary'):
            return circ.get_unitary()
        if hasattr(circ, 'to_matrix'):
            return circ.to_matrix()
        sim = Aer.get_backend('unitary_simulator')
        circ = transpile(circ, sim, optimization_level=1)  # simplify, but make no assumption about the initial state
        res = sim.run(circ).result()
        U   = res.get_unitary(decimals=decimals)
        if as_np:
            return np.array(U)
        return U

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
    # stubs
    def _qiskit_not_installed():
        raise ValueError("qiskit not installed! Use `pip install qiskit qiskit_aer pylatexenc`.")

    def run(circuit, shots=1, generate_state=True, plot=True, showqubits=None, showcoeff=True, showprobs=True, showrho=False, figsize=(16,4), title=""):
        _qiskit_not_installed()

    class exp_i:
        def __init__(self, H, k=1, use_pauli=False, trotter_steps=3):
            _qiskit_not_installed()

    def get_unitary(U):
        _qiskit_not_installed()

def get_pe_energies(U):
    if "qiskit" in sys.modules and isinstance(U, QuantumCircuit):
        U = get_unitary(U)
    eigvals = eig(U)[0]
    energies = np.angle(eigvals)/(2*np.pi)
    return energies

def show_eigenvecs(U, showrho=False):
    if "qiskit" in sys.modules and isinstance(U, QuantumCircuit):
        U = get_unitary(U)
    eigvecs = eig(U)[1]
    print(np.round(eigvecs, 3))
    for i in range(eigvecs.shape[1]):
        plotQ(eigvecs[:,i], figsize=(12,2), showrho=showrho)
    return eigvecs

def partial_operation(U, subsystem, env_state='0', check=1):
    """ Calculate the partial operation of a unitary U acting on a subsystem. Returns a set of Kraus operators. """
    n = count_qubits(U)
    subsystem = list(subsystem)
    q = len(subsystem)
    env = [i for i in range(n) if i not in subsystem]
    assert max(subsystem) <= n and min(subsystem) >= 0, f"Subsystem is has invalid qubits: {subsystem}"

    U = transpose_qubit_order(U, subsystem + env)
    U = U.reshape([2**q, 2**(n-q), 2**q, 2**(n-q)])

    env_state = ket(env_state, n-q, check=check)

    # decompose the resulting state into Kraus operators
    Kraus = []
    for i in range(2**(n-q)):
        Kraus.append(np.tensordot(U[:, i, :, :], env_state, axes=(2, 0)))
    return Kraus
