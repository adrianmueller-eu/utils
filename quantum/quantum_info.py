import sys
import numpy as np

from .state import count_qubits, partial_trace, op, ket, dm, ev
from ..mathlib import trace_norm
from ..prob import entropy

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
    n_A = count_qubits(observable_A)
    n_B = count_qubits(observable_B)
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