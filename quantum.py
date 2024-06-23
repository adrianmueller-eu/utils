import psutil
import numpy as np
import itertools
from functools import reduce
import matplotlib.pyplot as plt
import scipy.sparse as sp
from .mathlib import *
from .plot import colorize_complex
from .utils import duh
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
II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ = [np.kron(g1, g2) for g1, g2 in itertools.product([I,X,Y,Z], repeat=2)]

def C_(A):
    if not hasattr(A, 'shape'):
        A = np.array(A, dtype=complex)
    n = int(np.log2(A.shape[0]))
    return np.kron([[1,0],[0,0]], I_(n)) + np.kron([[0,0],[0,1]], A)
CNOT = CX = C_(X) # 0.5*(II + ZI - ZX + IX)
Toffoli = C_(C_(X))
SWAP = np.array([ # 0.5*(XX + YY + ZZ + II), CNOT @ r(reverse_qubit_order(CNOT)) @ CNOT
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

def Fourier_matrix(n, n_is_qubits=True):
    """Calculate the Fourier matrix of size `n`. The Fourier matrix is the matrix representation of the quantum Fourier transform (QFT)."""
    if n_is_qubits:
        n = 2**n
    return 1/np.sqrt(n) * np.array([[np.exp(2j*np.pi*i*j/n) for j in range(n)] for i in range(n)], dtype=complex)

def parse_unitary(unitary):
    """Parse a string representation of a unitary into its matrix representation. The result is guaranteed to be unitary.

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
        # think of XCXC, which applies X on the first and fifth qubit, controlled on the second and forth qubit
        # select the first "C", then recursively call s on the part before and after (if they exist), and combine them afterwards
        chunk_matrix = np.array([1]) # initialize with a stub
        for i, c in enumerate(chunk):
            if c == "C":
                n_before = i
                n_after = len(chunk) - i - 1
                # if it's the first C, then there is no part before
                if n_before == 0:
                    return np.kron([[1,0],[0,0]], I_(n_after)) + np.kron([[0,0],[0,1]], s(chunk[i+1:]))
                # if it's the last C, then there is no part after
                elif n_after == 0:
                    return np.kron(I_(n_before), [[1,0],[0,0]]) + np.kron(chunk_matrix, [[0,0],[0,1]])
                # if it's in the middle, then there is a part before and after
                else:
                    return np.kron(I_(n_before), np.kron([[1,0],[0,0]], I_(n_after))) + np.kron(chunk_matrix, np.kron([[0,0],[0,1]], s(chunk[i+1:])))
            # N is negative control, so it's the same as C, but with the roles of 0 and 1 reversed
            elif c == "N":
                n_before = i
                n_after = len(chunk) - i - 1
                # if it's the first N, then there is no part before
                if n_before == 0:
                    return np.kron([[0,0],[0,1]], I_(n_after)) + np.kron([[1,0],[0,0]], s(chunk[i+1:]))
                # if it's the last N, then there is no part after
                elif n_after == 0:
                    return np.kron(I_(n_before), [[0,0],[0,1]]) + np.kron(chunk_matrix, [[1,0],[0,0]])
                # if it's in the middle, then there is a part before and after
                else:
                    return np.kron(I_(n_before), np.kron([[0,0],[0,1]], I_(n_after))) + np.kron(chunk_matrix, np.kron([[1,0],[0,0]], s(chunk[i+1:])))
            # if there is no C, then it's just a single gate
            elif c == "T":
                 gate = T_gate
            else:
                 gate = globals()[chunk[i]]
            chunk_matrix = np.kron(chunk_matrix, gate)
        return chunk_matrix

    # Remove whitespace
    unitary = unitary.replace(" ", "")

    # Parse the unitary
    chunks = unitary.split("@")
    # Remove empty chunks
    chunks = [c for c in chunks if c != ""]
    # Use the first chunk to determine the number of qubits
    n = len(chunks[0])

    U = np.eye(2**n, dtype=complex)
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

    assert np.allclose(U @ U.conj().T, np.eye(2**n)), f"Result is not unitary: {U, U @ U.conj().T}"

    return U

try:
    ##############
    ### Qiskit ###
    ##############

    from qiskit import transpile
    from qiskit_aer import Aer
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit.quantum_info.operators import Operator

    # Other useful imports
    from qiskit.quantum_info import Statevector
    from qiskit.visualization import plot_histogram
    from qiskit.circuit.library import UnitaryGate

    def run(circuit, shots=1, generate_state=True, plot=True, showqubits=None, showcoeff=True, showprobs=True, showrho=False, figsize=(16,4), title=""):
        if shots > 10:
            tc = time_complexity(circuit)
            print("TC: %d, expected running time: %.3fs" % (tc, tc * 0.01))
        if generate_state:
            simulator = Aer.get_backend("statevector_simulator")
            if shots is None or shots == 0:
                print("Warning: shots=0 is not supported for statevector_simulator. Using shots=1 instead.")
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
        def __init__(self, H):
            self.H = H
            self.n = int(np.log2(len(self.H)))
            super().__init__(self.n)       # circuit on n qubits
            u = matexp(1j*self.H)          # create unitary from hamiltonian
            self.all_qubits = list(range(self.n))
            self.unitary(Operator(u), self.all_qubits, label="exp^iH") # add unitary to circuit

        def power(self, k):
            q_k = QuantumCircuit(self.n, name=f"exp({k}iH)")
            u = matexp(1j*k*self.H)
            q_k.unitary(Operator(u), self.all_qubits, label=f"exp^{k}iH")
            return q_k

    def get_unitary(circ, decimals=None):
        sim = Aer.get_backend('unitary_simulator')
        t_circuit = transpile(circ, sim)
        res = sim.run(t_circuit).result()
        return res.get_unitary(decimals=decimals)

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

    def time_complexity(qc, decompose_iterations=4, isTranspiled=False):
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

#############
### State ###
#############

def reverse_qubit_order(state):
    """So the last will be first, and the first will be last. Works for both, state vectors and density matrices."""
    state = np.array(state)
    n = int(np.log2(len(state)))

    # if vector, just reshape
    if len(state.shape) == 1 or state.shape[0] != state.shape[1]:
        return state.reshape([2]*n).T.flatten()
    elif state.shape[0] == state.shape[1]:
        return state.reshape([2,2]*n).T.reshape(2**n, 2**n).conj()

def partial_trace(rho, retain_qubits):
    """Trace out all qubits not specified in `retain_qubits`."""
    rho = np.array(rho)
    n = int(np.log2(rho.shape[0])) # number of qubits

    # pre-process retain_qubits
    if isinstance(retain_qubits, int):
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
    """This is a pervert version of the partial trace, but for state vectors. I'm not sure about the physical meaning of its output, but it was at times helpful to visualize and interpret subsystems, especially when the density matrix was out of reach (or better: out of memory)."""
    state = np.array(state)
    state[np.isnan(state)] = 0
    n = int(np.log2(len(state))) # nr of qubits

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

    state = state.flatten()
    state = normalize(state) # renormalize

    probs = probs.flatten()
    assert np.abs(np.sum(probs) - 1) < 1e-5, np.sum(probs) # sanity check

    return state, probs

def plotQ(state, showqubits=None, showcoeff=True, showprobs=True, showrho=False, figsize=None, title=""):
    """My best attempt so far to visualize a state vector. Control with `showqubits` which subsystem you're interested in (`None` will show the whole state). `showcoeff` utilitzes `state_trace`, `showprobs` shows a pie chart of the probabilities when measured in the standard basis, and `showrho` gives a plt.imshow view on the corresponding density matrix."""

    def tobin(n, places):
        return ("{0:0" + str(places) + "b}").format(n)

    def plotcoeff(ax, state):
        n = int(np.log2(len(state))) # nr of qubits
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
        n = int(np.log2(len(state))) # nr of qubits
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
        n = int(np.log2(len(state))) # nr of qubits
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
        n = int(np.log2(len(state))) # nr of qubits
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

def ket(specification):
    """Convert a string or dictionary of strings and weights to a state vector. The string can be a binary number or a combination of binary numbers and weights. The weights will be normalized to 1."""
    # if a string is given, convert it to a dictionary
    if isinstance(specification, (np.ndarray, list, tuple)):
        n = int(np.log2(len(specification)))
        assert len(specification) == 2**n, f"State vector has wrong length: {len(specification)} is not a power of 2!"
        return normalize(specification)
    elif type(specification) == str:
        # handle some special cases: |+>, |->, |i>, |-i>
        if specification == "+":
            return normalize(np.array([1,1], dtype=complex))
        elif specification == "-":
            return normalize(np.array([1,-1], dtype=complex))
        elif specification == "i":
            return normalize(np.array([1,1j], dtype=complex))
        elif specification == "-i":
            return normalize(np.array([1,-1j], dtype=complex))
        elif specification.isdigit() and len(specification.replace('0', '').replace('1', '')) > 0:
            a = int(specification)
            n = int(np.ceil(np.log2(a+1)))
            return normalize(bincoll_from_int(2**a, 2**n))

        # remove whitespace
        specification = specification.replace(" ", "")
        specification_dict = dict()
        n = None

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
                if n is not None:
                    assert len(state) == n, f"Part of the specification has wrong length: len('{state}') != {n}"
                else:
                    n = len(state)
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

def unket(state, as_dict=False):
    """ Reverse of above. The output is always guaranteed to be normalized.

    Example:
    >>> unket(ket('00+01+10+11'))
    '0.5*(00+01+10+11)'
    >>> unket(ket('00+01+10+11'), as_dict=True)
    {'00': 0.5, '01': 0.5, '10': 0.5, '11': 0.5}
    """
    state = normalize(state)
    n = int(np.log2(len(state)))
    if as_dict:
        # cast to float if imaginary part is zero
        if np.allclose(state.imag, 0):
            state = state.real
        return {binstr_from_int(i, n): state[i] for i in range(2**n) if state[i] != 0}

    # group by weight
    weights = {}
    for i in range(2**n):
        if state[i] != 0:
            weight = state[i]
            # Find a close value in weights
            for w in weights:
                if np.allclose(w, weight):
                    weight = w
                    break
            else:
                # remove imaginary part if it's zero
                if np.abs(weight.imag) < 1e-15:
                    weight = weight.real
                # create new weight
                weights[weight] = []
            # add state to weight
            weights[weight].append(binstr_from_int(i, n))

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

def op(specification1, specification2=None):
    # If it's already a matrix, ensure it's a density matrix and return it
    if type(specification1) != str:
        specification1 = np.array(specification1, copy=False)
        if len(specification1.shape) > 1:
            sp1_trace = np.trace(specification1)
            # trace normalize it if it's not already
            if not np.allclose(sp1_trace, 1):
                specification1 = specification1 / sp1_trace
            assert is_dm(specification1), f"The given matrix is not a density matrix!"
            return specification1
    s1 = ket(specification1)
    s2 = s1 if specification2 is None else ket(specification2)
    return np.outer(s1, s2.conj())

def dm(specification1, specification2=None):
    return op(specification1, specification2)

def probs(state):
    """Calculate the probabilities of measuring a state vector in the standard basis."""
    return np.abs(state)**2

def entropy_von_Neumann(state):
    """Calculate the von Neumann entropy of a given density matrix."""
    if not(isinstance(state, np.ndarray) and len(state.shape) == 2 and state.shape[0] == state.shape[1]):
        state = op(state)
    # S = -np.trace(state @ matlog(state)/np.log(2))
    # assert np.allclose(S.imag, 0), f"WTF: Entropy is not real: {S}"
    # return np.max(S.real, 0)  # fix rounding errors
    eigs = np.linalg.eigvalsh(state)
    assert np.allclose(np.sum(eigs), 1), f"Density matrix is not normalized! {np.sum(eigs)}"
    assert np.all(eigs >= -sys.float_info.epsilon), f"Density matrix is not positive semidefinite! {eigs}"
    return entropy(eigs)

def entropy_entanglement(state, subsystem_qubits):
    """Calculate the entanglement entropy of a quantum state (density matrix or vector) with respect to the given subsystem."""
    return entropy_von_Neumann(partial_trace(state, subsystem_qubits))

def is_dm(rho):
    """Check if matrix `rho` is a density matrix."""
    rho = np.array(rho)
    return np.allclose(np.trace(rho), 1) and is_psd(rho)

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

def Schmidt_decomposition(state, subsystem_qubits):
    """Calculate the Schmidt decomposition of a pure state with respect to the given subsystem."""
    state = np.array(state)
    assert len(state.shape) == 1, f"State must be a vector, but has shape: {state.shape}"
    n = int(np.log2(len(state)))
    assert len(subsystem_qubits) <= n-1, f"Too many subsystem qubits: {len(subsystem_qubits)} >= {n}"

    # reorder the qubits so that the subsystem qubits are at the beginning
    subsystem_qubits = list(subsystem_qubits)
    other_qubits = sorted(set(range(n)) - set(subsystem_qubits))
    state = state.reshape([2]*n).transpose(subsystem_qubits + other_qubits).flatten()

    # calculate the Schmidt coefficients and basis using SVD
    a_jk = state.reshape([2**len(subsystem_qubits), 2**len(other_qubits)])
    U, S, V = np.linalg.svd(a_jk)
    return S, U.T, V

def gibbs(H, beta=1):
    """Calculate the Gibbs state of a Hamiltonian `H` at inverse temperature `beta`."""
    H = np.array(H)
    assert is_hermitian(H), "Hamiltonian must be Hermitian!"
    assert beta >= 0, "Inverse temperature must be positive!"
    E, U = np.linalg.eigh(H)
    E = softmax(E, -beta)
    return U @ np.diag(E) @ U.conj().T

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

def ceil_state(A, eps=1e-6):
    """ Calculate the ground state using the power iteration algorithm. """
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
            print("Warning: Hamiltonian is a scalar!")

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
            print(f"Warning: Using a dense matrix for a {n}-qubit Hamiltonian is not recommended. Use sparse=True.")
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
        assert np.allclose(H, H.conj().T), f"The given Hamiltonian {hamiltonian} is not Hermitian: {H}"

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

def ising(n_qubits, J=(-1,1), h=(-1,1), g=(-1,1), offset=0, kind='1d', circular=False):
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
            raise ValueError(f"For kind={kind}, n_qubits must be a tuple of length 2, but is {n_qubits}")
        couplings = (n_total_qubits, n_total_qubits)
    elif kind == '3d':
        if np.isscalar(n_qubits) or len(n_qubits) != 3:
            raise ValueError(f"For kind={kind}, n_qubits must be a tuple of length 3, but is {n_qubits}")
        couplings = (n_total_qubits, n_total_qubits)
    elif kind == 'pairwise':
        assert type(n_qubits) == int, f"For kind={kind}, n_qubits must be an integer, but is {n_qubits}"
        couplings = (n_qubits, n_qubits)
    elif kind == 'all':
        assert type(n_qubits) == int, f"For kind={kind}, n_qubits must be an integer, but is {n_qubits}"
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

    # generate the Hamiltonian
    H_str = ''
    # pairwise interactions
    if kind == '1d':
        if np.isscalar(J):
            for i in range(n_qubits-1):
                H_str += 'I'*i + 'ZZ' + 'I'*(n_qubits-i-2) + ' + '
            # last and first qubit
            if circular and n_qubits > 2:
                H_str += 'Z' + 'I'*(n_qubits-2) + 'Z' + ' + '
        else:
            for i in range(n_qubits-1):
                if J[i] != 0:
                    H_str += str(J[i]) + '*' + 'I'*i + 'ZZ' + 'I'*(n_qubits-i-2) + ' + '
            # last and first qubit
            if circular and n_qubits > 2 and J[n_qubits-1] != 0:
                H_str += str(J[n_qubits-1]) + '*' + 'Z' + 'I'*(n_qubits-2) + 'Z' + ' + '
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
                        first_index = min(index_node, index_neighbor)
                        second_index = max(index_node, index_neighbor)
                        if not np.isscalar(J):
                            if J[first_index, second_index] == 0:
                                continue
                            H_str += str(J[first_index, second_index]) + '*'
                        H_str += 'I'*first_index + 'Z' + 'I'*(second_index-first_index-1) + 'Z' + 'I'*(n_qubits[0]*n_qubits[1]-second_index-1) + ' + '
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
                            first_index = min(index_node, index_neighbor)
                            second_index = max(index_node, index_neighbor)
                            if not np.isscalar(J):
                                if J[first_index, second_index] == 0:
                                    continue
                                H_str += str(J[first_index, second_index]) + '*' 
                            H_str += 'I'*first_index + 'Z' + 'I'*(second_index-first_index-1) + 'Z' + 'I'*(n_qubits[0]*n_qubits[1]*n_qubits[2]-second_index-1) + ' + '
    elif kind == 'pairwise':
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                if not np.isscalar(J):
                    if J[i,j] == 0:
                        continue
                    H_str += str(J[i,j]) + '*'
                H_str += 'I'*i + 'Z' + 'I'*(j-i-1) + 'Z' + 'I'*(n_qubits-j-1) + ' + '
    elif kind == 'all':
        if np.isscalar(J):
            if n_qubits > 20:
                raise ValueError("Printing out all interactions for n_qubits > 20 is not recommended. Please use a dict instead.")
            for i in range(2, n_qubits+1):
                for membership in itertools.combinations(range(n_qubits), i):
                    H_str += ''.join(['Z' if j in membership else 'I' for j in range(n_qubits)]) + ' + '
        else: # J is a dict of tuples of qubit indices to interaction strengths
            for membership, strength in J.items():
                if strength == 0:
                    continue
                H_str += str(strength) + '*' + ''.join(['Z' if j in membership else 'I' for j in range(n_qubits)]) + ' + '
    else:
        raise ValueError(f"Unknown kind {kind}")

    if np.isscalar(J) and n_total_qubits > 1:
        if J != 0:
            H_str = str(J) + '*(' + H_str[:-3] + ') + '
        else:
            H_str = ''

    # local longitudinal fields
    if np.any(h):
        if np.isscalar(h):
            H_str += str(h) + '*(' + ' + '.join(['I'*i + 'Z' + 'I'*(n_total_qubits-i-1) for i in range(n_total_qubits)]) + ') + '
        else:
            H_str += ' + '.join([str(h[i]) + '*' + 'I'*i + 'Z' + 'I'*(n_total_qubits-i-1) for i in range(n_total_qubits) if h[i] != 0]) + ' + '
    # local transverse fields
    if np.any(g):
        if np.isscalar(g):
            H_str += str(g) + '*(' + ' + '.join(['I'*i + 'X' + 'I'*(n_total_qubits-i-1) for i in range(n_total_qubits)]) + ') + '
        else:
            H_str += ' + '.join([str(g[i]) + '*' + 'I'*i + 'X' + 'I'*(n_total_qubits-i-1) for i in range(n_total_qubits) if g[i] != 0]) + ' + '
    # Add I*n - I*n if there are no fields
    if H_str == '':
        H_str += 'I'*n_total_qubits + ' - ' + 'I'*n_total_qubits + ' + '
    # offset
    if np.any(offset):
        H_str += str(offset)
    else:
        H_str = H_str[:-3]
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
def pauli_decompose(H, include_zero=False):
    r"""Decomposes a Hermitian matrix into a linear combination of Pauli operators.

    Parameters
        H (ndarray): a Hermitian matrix of shape ``(2**n, 2**n)``
        include_zero (bool): if True, include Pauli terms with zero coefficients in the decomposition

    Returns
        tuple[list[float], list[str]]: the coefficients and the Pauli operator strings

    Example
    >>> H = np.array([[-2, -2+1j, -2, -2], [-2-1j,  0,  0, -1], [-2,  0, -2, -1], [-2, -1, -1,  0]])
    >>> pauli_decompose(H)
    ([-1.0, -1.5, -0.5, -1.0, -1.5, -1.0, -0.5, 1.0, -0.5, -0.5],
     ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XZ', 'YY', 'ZX', 'ZY'])
    """
    n = int(np.log2(len(H)))
    N = 2**n

    if H.shape != (N, N):
        raise ValueError(f"The matrix should have shape (2**n, 2**n), for any qubit number n>=1, but is {H.shape}")

    if not np.allclose(H, H.conj().T):
        raise ValueError("The matrix is not Hermitian")

    obs_lst = []
    coeffs = []

    for term, basis_matrix in zip(pauli_basis(n, kind='str'), pauli_basis(n, kind='np')):
        coeff = np.trace(basis_matrix @ H) / N  # project H onto the basis matrix
        coeff = np.real_if_close(coeff).item()

        if not np.allclose(coeff, 0) or include_zero:
            coeffs.append(coeff)
            obs_lst.append(term)

    return coeffs, obs_lst

###############
### Aliases ###
###############

ph = parse_hamiltonian
pu = parse_unitary

#############
### Tests ###
#############

def test_quantum_all():
    tests = [
        _test_constants,
        _test_is_dm,
        _test_random_ket,
        _test_random_dm,
        _test_random_ham,
        _test_get_H_energies_eq_get_pe_energies,
        _test_ph,
        _test_ground_state,
        _test_reverse_qubit_order,
        _test_partial_trace,
        _test_entropy_von_Neumann,
        _test_entropy_entanglement,
        _test_fidelity,
        _test_Schmidt_decomposition,
        _test_ising,
        _test_pauli_basis,
        _test_pauli_decompose
    ]

    for test in tests:
        print("Running", test.__name__, "... ", end="", flush=True)
        test()
        print("Test succeed!", flush=True)

def _test_constants():
    global I, X, Y, Z, H_gate, S, T_gate, CNOT, SWAP
    H = H_gate

    assert is_involutory(X)
    assert is_involutory(Y)
    assert is_involutory(Z)
    assert is_involutory(H)
    assert np.allclose(S @ S, Z)
    assert np.allclose(T_gate @ T_gate, S)
    assert is_involutory(CNOT)
    assert is_involutory(SWAP)
    assert np.allclose(Rx(2*np.pi), -I)
    assert np.allclose(Ry(2*np.pi), -I)
    assert np.allclose(Rz(2*np.pi), -I)

def _test_is_dm():
    assert is_dm(np.eye(2**2)/2**2)
    # random Bloch vectors
    for _ in range(100):
        v = np.random.uniform(-1, 1, 3)
        if np.linalg.norm(v) > 1:
            v = normalize(v)
        # create dm from Bloch vector
        rho = (I + v[0]*X + v[1]*Y + v[2]*Z)/2
        assert is_dm(rho)

def _test_random_ket():
    for _ in range(100):
        n_qubits = np.random.randint(1, 10)
        psi = random_ket(n_qubits)
        assert psi.shape == (2**n_qubits,)
        assert np.allclose(np.linalg.norm(psi), 1)

def _test_random_dm():
    for _ in range(100):
        n_qubits = np.random.randint(1, 5)
        rho = random_dm(n_qubits)
        assert is_dm(rho)

def _test_ph():
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

def _test_ground_state():
    H = parse_hamiltonian('ZZII + IZZI + IIZZ', dtype=float)
    ge, gs = -3, ket('0101')

    res_exact = ground_state_exact(H)
    assert np.allclose(res_exact[0], ge)
    assert np.allclose(res_exact[1], gs)

    res_ITE = ground_state_ITE(H)
    assert np.allclose(res_ITE[0], ge)
    # assert np.allclose(res_ITE[1], gs) # might be complex due to random state initialization

def _test_get_H_energies_eq_get_pe_energies():
    n_qubits = np.random.randint(1, 5)
    n_terms = np.random.randint(1, 100)
    n_terms = min(n_terms, 2**(2*n_qubits)-1)
    H = random_ham(n_qubits, n_terms, scaling=False)
    H = parse_hamiltonian(H)

    A = np.sort(get_pe_energies(exp_i(H)))
    B = np.sort(get_H_energies(H, expi=True))
    assert np.allclose(A, B), f"{A} ≠ {B}"

def _test_reverse_qubit_order():
    # known 3-qubit matrix
    psi = np.kron(np.kron([1,1], [0,1]), [1,-1])
    psi_rev = np.kron(np.kron([1,-1], [0,1]), [1,1])
    psi_rev2 = reverse_qubit_order(psi)
    assert np.allclose(psi_rev, psi_rev2)

    # same as above, but with n random qubits
    n = 10
    psis = [random_ket(1) for _ in range(n)]
    psi = psis[0]
    for i in range(1,n):
        psi = np.kron(psi, psis[i])
    psi_rev = psis[-1]
    for i in range(1,n):
        psi_rev = np.kron(psi_rev, psis[-i-1])

    psi_rev2 = reverse_qubit_order(psi)
    assert np.allclose(psi_rev, psi_rev2)

    # general hamiltonian
    H = parse_hamiltonian('IIIXX')
    H_rev = parse_hamiltonian('XXIII')
    H_rev2 = reverse_qubit_order(H)
    assert np.allclose(H_rev, H_rev2)

    H = parse_hamiltonian('XI + YI')
    H_rev = parse_hamiltonian('IX + IY')
    H_rev2 = reverse_qubit_order(H)
    assert np.allclose(H_rev, H_rev2), f"{H_rev} \n≠\n {H_rev2}"

    # pure density matrix
    psi = np.kron(np.kron([1,1], [0,1]), [1,-1])
    rho = np.outer(psi, psi)
    psi_rev = np.kron(np.kron([1,-1], [0,1]), [1,1])
    rho_rev = np.outer(psi_rev, psi_rev)
    rho_rev2 = reverse_qubit_order(rho)
    assert np.allclose(rho_rev, rho_rev2)

    # draw n times 2 random 1-qubit states and a probability distribution over all n pairs
    n = 10
    psis = [[random_dm(1) for _ in range(2)] for _ in range(n)]
    p = normalize(np.random.rand(n), p=1)
    # compute the average state
    psi = np.zeros((2**2, 2**2), dtype=complex)
    for i in range(n):
        psi += p[i]*np.kron(psis[i][0], psis[i][1])
    # compute the average state with reversed qubit order
    psi_rev = np.zeros((2**2, 2**2), dtype=complex)
    for i in range(n):
        psi_rev += p[i]*np.kron(psis[i][1], psis[i][0])

    psi_rev2 = reverse_qubit_order(psi)
    assert np.allclose(psi_rev, psi_rev2), f"psi_rev = {psi_rev}\npsi_rev2 = {psi_rev2}"

def _test_partial_trace():
    # known 4x4 matrix
    rho = np.arange(16).reshape(4,4)
    rhoA_expected = np.array([[ 5, 9], [21, 25]])
    rhoA_actual   = partial_trace(rho, 0)
    assert np.allclose(rhoA_expected, rhoA_actual), f"rho_expected = {rhoA_expected}\nrho_actual = {rhoA_actual}"

    # two separable density matrices
    rhoA = random_dm(2)
    rhoB = random_dm(3)
    rho = np.kron(rhoA, rhoB)
    rhoA_expected = rhoA
    rhoA_actual   = partial_trace(rho, [0,1])
    assert np.allclose(rhoA_expected, rhoA_actual), f"rho_expected = {rhoA_expected}\nrho_actual = {rhoA_actual}"

    # two separable state vectors
    psiA = random_ket(2)
    psiB = random_ket(3)
    psi = np.kron(psiA, psiB)
    psiA_expected = np.outer(psiA, psiA.conj())
    psiA_actual   = partial_trace(psi, [0,1])
    assert np.allclose(psiA_expected, psiA_actual), f"psi_expected = {psiA_expected}\npsi_actual = {psiA_actual}"

    # total trace
    st = random_ket(3)
    st_tr = partial_trace(st, [])
    assert np.allclose(np.array([[1]]), st_tr), f"st_tr = {st_tr} ≠ 1"
    rho = random_dm(3)
    rho_tr = partial_trace(rho, [])
    assert np.allclose(np.array([[1]]), rho_tr), f"rho_expected = {rhoA_expected}\nrho_actual = {rhoA_actual}"

    # retain all qubits
    st = random_ket(3)
    st_tr = partial_trace(st, [0,1,2])
    st_expected = np.outer(st, st.conj())
    assert st_expected.shape == st_tr.shape, f"st_expected.shape = {st_expected.shape} ≠ st_tr.shape = {st_tr.shape}"
    assert np.allclose(st_expected, st_tr), f"st_expected = {st_expected} ≠ st_tr = {st_tr}"
    rho = random_dm(2)
    rho_tr = partial_trace(rho, [0,1])
    assert rho.shape == rho_tr.shape, f"rho.shape = {rho.shape} ≠ rho_tr.shape = {rho_tr.shape}"
    assert np.allclose(rho, rho_tr), f"rho_expected = {rhoA_expected}\nrho_actual = {rhoA_actual}"

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
    rho_expected = partial_trace(np.outer(psi, psi.conj()), subsystem)
    rho_actual = np.sum([l_i**2 * np.outer(A_i, A_i.conj()) for l_i, A_i in zip(l, A)], axis=0)
    assert np.allclose(rho_expected, rho_actual), f"rho_expected - rho_actual = {rho_expected - rho_actual}"

    # check RDM for subsystem B
    rho_expected = partial_trace(np.outer(psi, psi.conj()), [i for i in range(n) if i not in subsystem])
    rho_actual = np.sum([l_i**2 * np.outer(B_i, B_i.conj()) for l_i, B_i in zip(l, B)], axis=0)
    assert np.allclose(rho_expected, rho_actual), f"rho_expected - rho_actual = {rho_expected - rho_actual}"

    # check entanglement entropy
    S_expected = entropy_entanglement(psi, subsystem)
    S_actual = -np.sum([l_i**2 * np.log2(l_i**2) for l_i in l])
    assert np.allclose(S_expected, S_actual), f"S_expected = {S_expected} ≠ S_actual = {S_actual}"


def _test_ising():
    # 1d
    H_str = ising(5, J=1.5, h=0, g=0, offset=0, kind='1d', circular=False)
    expected = "1.5*(ZZIII + IZZII + IIZZI + IIIZZ)"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    H_str = ising(3, J=0, h=3, g=2)
    expected = '3*(ZII + IZI + IIZ) + 2*(XII + IXI + IIX)'
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    H_str = ising(5, J=1.5, h=1.1, g=0.5, offset=0.5, kind='1d', circular=True)
    expected = "1.5*(ZZIII + IZZII + IIZZI + IIIZZ + ZIIIZ) + 1.1*(ZIIII + IZIII + IIZII + IIIZI + IIIIZ) + 0.5*(XIIII + IXIII + IIXII + IIIXI + IIIIX) + 0.5"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    H_str = ising(3, J=[0.6,0.7,0.8], h=[0.1,0.2,0.7], g=[0.6,0.1,1.5], offset=0.5, kind='1d', circular=True)
    expected = "0.6*ZZI + 0.7*IZZ + 0.8*ZIZ + 0.1*ZII + 0.2*IZI + 0.7*IIZ + 0.6*XII + 0.1*IXI + 1.5*IIX + 0.5"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    H_str = ising(3, J=[0,1], h=[1,2], g=[2,5], offset=0.5, kind='1d', circular=True)
    # random, but count terms in H_str instead
    n_terms = len(H_str.split('+'))
    assert n_terms == 10, f"n_terms = {n_terms}\nexpected = 10"

    # 2d
    H_str = ising((2,2), J=1.5, h=0, g=0, offset=0, kind='2d', circular=False)
    expected = "1.5*(ZIZI + ZZII + IZIZ + IIZZ)"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    H_str = ising((3,3), J=1.5, h=1.1, g=0.5, offset=0.5, kind='2d', circular=True)
    expected = "1.5*(ZIIZIIIII + ZZIIIIIII + IZIIZIIII + IZZIIIIII + IIZIIZIII + ZIZIIIIII + IIIZIIZII + IIIZZIIII + IIIIZIIZI + IIIIZZIII + IIIIIZIIZ + IIIZIZIII + IIIIIIZZI + ZIIIIIZII + IIIIIIIZZ + IZIIIIIZI + IIZIIIIIZ + IIIIIIZIZ) + 1.1*(ZIIIIIIII + IZIIIIIII + IIZIIIIII + IIIZIIIII + IIIIZIIII + IIIIIZIII + IIIIIIZII + IIIIIIIZI + IIIIIIIIZ) + 0.5*(XIIIIIIII + IXIIIIIII + IIXIIIIII + IIIXIIIII + IIIIXIIII + IIIIIXIII + IIIIIIXII + IIIIIIIXI + IIIIIIIIX) + 0.5"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    # 3d
    H_str = ising((2,2,3), kind='3d', J=1.8, h=0, g=0, offset=0, circular=False)
    expected = "1.8*(ZIIIIIZIIIII + ZIIZIIIIIIII + ZZIIIIIIIIII + IZIIIIIZIIII + IZIIZIIIIIII + IZZIIIIIIIII + IIZIIIIIZIII + IIZIIZIIIIII + IIIZIIIIIZII + IIIZZIIIIIII + IIIIZIIIIIZI + IIIIZZIIIIII + IIIIIZIIIIIZ + IIIIIIZIIZII + IIIIIIZZIIII + IIIIIIIZIIZI + IIIIIIIZZIII + IIIIIIIIZIIZ + IIIIIIIIIZZI + IIIIIIIIIIZZ)"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    H_str = ising((2,2,3), kind='3d', J=1.2, h=1.5, g=2, offset=0, circular=True)
    expected = "1.2*(ZIIIIIZIIIII + ZIIZIIIIIIII + ZZIIIIIIIIII + IZIIIIIZIIII + IZIIZIIIIIII + IZZIIIIIIIII + IIZIIIIIZIII + IIZIIZIIIIII + ZIZIIIIIIIII + IIIZIIIIIZII + IIIZZIIIIIII + IIIIZIIIIIZI + IIIIZZIIIIII + IIIIIZIIIIIZ + IIIZIZIIIIII + IIIIIIZIIZII + IIIIIIZZIIII + IIIIIIIZIIZI + IIIIIIIZZIII + IIIIIIIIZIIZ + IIIIIIZIZIII + IIIIIIIIIZZI + IIIIIIIIIIZZ + IIIIIIIIIZIZ) + 1.5*(ZIIIIIIIIIII + IZIIIIIIIIII + IIZIIIIIIIII + IIIZIIIIIIII + IIIIZIIIIIII + IIIIIZIIIIII + IIIIIIZIIIII + IIIIIIIZIIII + IIIIIIIIZIII + IIIIIIIIIZII + IIIIIIIIIIZI + IIIIIIIIIIIZ) + 2*(XIIIIIIIIIII + IXIIIIIIIIII + IIXIIIIIIIII + IIIXIIIIIIII + IIIIXIIIIIII + IIIIIXIIIIII + IIIIIIXIIIII + IIIIIIIXIIII + IIIIIIIIXIII + IIIIIIIIIXII + IIIIIIIIIIXI + IIIIIIIIIIIX)"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    H_str = ising((3,3,3), kind='3d', J=1.5, h=0, g=0, offset=0, circular=True)
    expected = "1.5*(ZIIIIIIIIZIIIIIIIIIIIIIIIII + ZIIZIIIIIIIIIIIIIIIIIIIIIII + ZZIIIIIIIIIIIIIIIIIIIIIIIII + IZIIIIIIIIZIIIIIIIIIIIIIIII + IZIIZIIIIIIIIIIIIIIIIIIIIII + IZZIIIIIIIIIIIIIIIIIIIIIIII + IIZIIIIIIIIZIIIIIIIIIIIIIII + IIZIIZIIIIIIIIIIIIIIIIIIIII + ZIZIIIIIIIIIIIIIIIIIIIIIIII + IIIZIIIIIIIIZIIIIIIIIIIIIII + IIIZIIZIIIIIIIIIIIIIIIIIIII + IIIZZIIIIIIIIIIIIIIIIIIIIII + IIIIZIIIIIIIIZIIIIIIIIIIIII + IIIIZIIZIIIIIIIIIIIIIIIIIII + IIIIZZIIIIIIIIIIIIIIIIIIIII + IIIIIZIIIIIIIIZIIIIIIIIIIII + IIIIIZIIZIIIIIIIIIIIIIIIIII + IIIZIZIIIIIIIIIIIIIIIIIIIII + IIIIIIZIIIIIIIIZIIIIIIIIIII + IIIIIIZZIIIIIIIIIIIIIIIIIII + ZIIIIIZIIIIIIIIIIIIIIIIIIII + IIIIIIIZIIIIIIIIZIIIIIIIIII + IIIIIIIZZIIIIIIIIIIIIIIIIII + IZIIIIIZIIIIIIIIIIIIIIIIIII + IIIIIIIIZIIIIIIIIZIIIIIIIII + IIZIIIIIZIIIIIIIIIIIIIIIIII + IIIIIIZIZIIIIIIIIIIIIIIIIII + IIIIIIIIIZIIIIIIIIZIIIIIIII + IIIIIIIIIZIIZIIIIIIIIIIIIII + IIIIIIIIIZZIIIIIIIIIIIIIIII + IIIIIIIIIIZIIIIIIIIZIIIIIII + IIIIIIIIIIZIIZIIIIIIIIIIIII + IIIIIIIIIIZZIIIIIIIIIIIIIII + IIIIIIIIIIIZIIIIIIIIZIIIIII + IIIIIIIIIIIZIIZIIIIIIIIIIII + IIIIIIIIIZIZIIIIIIIIIIIIIII + IIIIIIIIIIIIZIIIIIIIIZIIIII + IIIIIIIIIIIIZIIZIIIIIIIIIII + IIIIIIIIIIIIZZIIIIIIIIIIIII + IIIIIIIIIIIIIZIIIIIIIIZIIII + IIIIIIIIIIIIIZIIZIIIIIIIIII + IIIIIIIIIIIIIZZIIIIIIIIIIII + IIIIIIIIIIIIIIZIIIIIIIIZIII + IIIIIIIIIIIIIIZIIZIIIIIIIII + IIIIIIIIIIIIZIZIIIIIIIIIIII + IIIIIIIIIIIIIIIZIIIIIIIIZII + IIIIIIIIIIIIIIIZZIIIIIIIIII + IIIIIIIIIZIIIIIZIIIIIIIIIII + IIIIIIIIIIIIIIIIZIIIIIIIIZI + IIIIIIIIIIIIIIIIZZIIIIIIIII + IIIIIIIIIIZIIIIIZIIIIIIIIII + IIIIIIIIIIIIIIIIIZIIIIIIIIZ + IIIIIIIIIIIZIIIIIZIIIIIIIII + IIIIIIIIIIIIIIIZIZIIIIIIIII + IIIIIIIIIIIIIIIIIIZIIZIIIII + IIIIIIIIIIIIIIIIIIZZIIIIIII + ZIIIIIIIIIIIIIIIIIZIIIIIIII + IIIIIIIIIIIIIIIIIIIZIIZIIII + IIIIIIIIIIIIIIIIIIIZZIIIIII + IZIIIIIIIIIIIIIIIIIZIIIIIII + IIIIIIIIIIIIIIIIIIIIZIIZIII + IIZIIIIIIIIIIIIIIIIIZIIIIII + IIIIIIIIIIIIIIIIIIZIZIIIIII + IIIIIIIIIIIIIIIIIIIIIZIIZII + IIIIIIIIIIIIIIIIIIIIIZZIIII + IIIZIIIIIIIIIIIIIIIIIZIIIII + IIIIIIIIIIIIIIIIIIIIIIZIIZI + IIIIIIIIIIIIIIIIIIIIIIZZIII + IIIIZIIIIIIIIIIIIIIIIIZIIII + IIIIIIIIIIIIIIIIIIIIIIIZIIZ + IIIIIZIIIIIIIIIIIIIIIIIZIII + IIIIIIIIIIIIIIIIIIIIIZIZIII + IIIIIIIIIIIIIIIIIIIIIIIIZZI + IIIIIIZIIIIIIIIIIIIIIIIIZII + IIIIIIIIIIIIIIIIIIZIIIIIZII + IIIIIIIIIIIIIIIIIIIIIIIIIZZ + IIIIIIIZIIIIIIIIIIIIIIIIIZI + IIIIIIIIIIIIIIIIIIIZIIIIIZI + IIIIIIIIZIIIIIIIIIIIIIIIIIZ + IIIIIIIIIIIIIIIIIIIIZIIIIIZ + IIIIIIIIIIIIIIIIIIIIIIIIZIZ)"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    # pairwise
    H_str = ising(4, J=-.5, h=.4, g=.7, offset=1, kind='pairwise')
    expected = "-0.5*(ZZII + ZIZI + ZIIZ + IZZI + IZIZ + IIZZ) + 0.4*(ZIII + IZII + IIZI + IIIZ) + 0.7*(XIII + IXII + IIXI + IIIX) + 1"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    # full
    H_str = ising(3, J=1.5, h=.4, g=.7, offset=1, kind='all')
    expected = "1.5*(ZZI + ZIZ + IZZ + ZZZ) + 0.4*(ZII + IZI + IIZ) + 0.7*(XII + IXI + IIX) + 1"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    H_str = ising(3, kind='all', J={(0,1): 2, (0,1,2): 3, (1,2):0}, g=0, h=1.35)
    expected = "2*ZZI + 3*ZZZ + 1.35*(ZII + IZI + IIZ)"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

    J_dict = {
        (0,1): 1.5,
        (0,2): 2,
        (1,2): 0.5,
        (0,1,2): 3,
        (0,1,2,3): 0.5
    }
    H_str = ising(4, J=J_dict, h=.3, g=.5, offset=1.2, kind='all')
    expected = "1.5*ZZII + 2*ZIZI + 0.5*IZZI + 3*ZZZI + 0.5*ZZZZ + 0.3*(ZIII + IZII + IIZI + IIIZ) + 0.5*(XIII + IXII + IIXI + IIIX) + 1.2"
    assert H_str == expected, f"\nH_str    = {H_str}\nexpected = {expected}"

def _test_pauli_basis():
    n = np.random.randint(1,4)
    pauli_n = pauli_basis(n)

    # check the number of generators
    n_expected = 2**(2*n)
    assert len(pauli_n) == n_expected, f"Number of generators is {len(pauli_n)}, but should be {n_expected}!"

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
    coeff, basis = pauli_decompose(np.eye(2**n), include_zero=True)
    n_expected = 2**(2*n)  # == len(pauli_basis(n))
    assert len(coeff) == n_expected, f"len(coeff) = {len(coeff)} ≠ {n_expected}"
    assert len(basis) == n_expected, f"len(basis) = {len(basis)} ≠ {n_expected}"