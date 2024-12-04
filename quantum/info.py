import sys
import numpy as np

from .state import count_qubits, partial_trace, ket, dm, ev
from ..mathlib import trace_norm, matsqrth_psd, is_hermitian
from ..prob import entropy

def von_neumann_entropy(state, check=True):
    """Calculate the von Neumann entropy of a given density matrix."""
    state = np.asarray(state)
    if len(state.shape) == 1:
        return 0  # pure state
    if check:
        assert is_hermitian(state), "Density matrix is not Hermitian!"
    eigs = np.linalg.eigvalsh(state)
    if check:
        assert abs(np.sum(eigs) - 1) < 1e-10, f"State is not normalized! {np.sum(eigs)}"  # dm trace-normalizes
        assert np.all(eigs >= -len(eigs)*sys.float_info.epsilon), f"Density matrix is not positive semidefinite: {eigs}"
    return entropy(eigs)

def entanglement_entropy(state, subsystem_qubits, check=True):
    """Calculate the entanglement entropy of a quantum state (density matrix or vector) with respect to the given subsystem."""
    return von_neumann_entropy(partial_trace(state, subsystem_qubits), check=check)

def mutual_information_quantum(state, subsystem_qubits, check=True):
    n = count_qubits(state)
    S_AB = von_neumann_entropy(state, check=check)
    S_A = entanglement_entropy(state, subsystem_qubits, check=False)
    B = [i for i in range(n) if i not in subsystem_qubits]
    S_B = entanglement_entropy(state, B, check=False)
    return S_A + S_B - S_AB

def fidelity(state1, state2):
    """Calculate the fidelity between two quantum states."""
    state1 = np.asarray(state1)
    state2 = np.asarray(state2)

    if len(state1.shape) == 1 and len(state2.shape) == 1:
        return np.abs(state1 @ state2.conj())**2
    elif len(state1.shape) == 2 and len(state2.shape) == 1:
        return np.abs(state2.conj() @ state1 @ state2)
    elif len(state1.shape) == 1 and len(state2.shape) == 2:
        return np.abs(state1.conj() @ state2 @ state1)
    elif len(state1.shape) == 2 and len(state2.shape) == 2:
        # state1_sqrt = matsqrt(state1)
        # return np.trace(matsqrt(state1_sqrt @ state2 @ state1_sqrt))**2  # textbook formula
        # return np.sum(np.sqrt(np.linalg.eigvals(state1 @ state2)))**2  # this is correct and faster
        state1_sqrt = matsqrth_psd(state1)
        state2_sqrt = matsqrth_psd(state2)
        S = np.linalg.svd(state1_sqrt @ state2_sqrt, compute_uv=False) # this is in between in efficiency, but the more stable
        return np.sum(S)**2
    else:
        raise ValueError(f"Can't calculate fidelity between {state1.shape} and {state2.shape}")

def trace_distance(rho1, rho2):
    """Calculate the trace distance between two density matrices."""
    rho1, rho2 = dm(rho1), dm(rho2)
    return 0.5 * trace_norm(rho1 - rho2)

def schmidt_decomposition(state, subsystem_qubits):
    """Calculate the Schmidt decomposition of a pure state with respect to the given subsystem."""
    state = np.asarray(state)
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
    state = np.asarray(state)
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