import warnings
import numpy as np
from functools import reduce

from .constants import I_, I, X, Y, Z, S, T_gate, H  # used in parse_unitary -> globals()
from .state import dm, count_qubits, plotQ
from .hamiltonian import pauli_decompose
from ..mathlib import is_unitary

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
                gate = dm(1)
            elif c == "N":
                gate = dm(0)
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
                    on, off = (dm(1), dm(0)) if c == "C" else (dm(0), dm(1))
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