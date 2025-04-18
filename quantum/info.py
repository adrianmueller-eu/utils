import sys
import numpy as np
import itertools
try:
    import scipy.sparse as sp
except ImportError:
    pass

from .state import count_qubits, partial_trace, op, dm, ev, as_state
from ..mathlib import trace_norm, matsqrth_psd, allclose0, eigvalsh, svd
from ..prob import entropy
from ..utils import is_iterable, is_from_assert, is_int

def von_neumann_entropy(state, check=2):
    """ Calculate the von Neumann entropy of a given density matrix. """
    state = as_state(state, check=check)
    if len(state.shape) == 1:
        return 0  # pure state
    eigs = eigvalsh(state)
    if check >= 1:
        assert np.all(eigs >= -len(eigs)*sys.float_info.epsilon), f"Density matrix is not positive semidefinite: {eigs[:5]} ..."
    return entropy(eigs)

def entanglement_entropy(state, subsystem_qubits, check=2):
    """ Calculate the entanglement entropy of a quantum state (density matrix or vector) with respect to the given subsystem. """
    return von_neumann_entropy(partial_trace(state, subsystem_qubits), check=check)

def mutual_information_quantum(state, subsystem_qubits, check=2):
    state = as_state(state, check=check)
    n = count_qubits(state)
    S_AB = von_neumann_entropy(state, check=check)
    S_A = entanglement_entropy(state, subsystem_qubits, check=0)
    B = [i for i in range(n) if i not in subsystem_qubits]
    S_B = entanglement_entropy(state, B, check=0)
    return S_A + S_B - S_AB

def purity(state):
    """ Calculate the purity of a quantum state. """
    if len(state.shape) == 1:
        return np.abs(state @ state.conj())
    elif len(state.shape) == 2:
        return np.trace(state @ state).real
    else:
        raise ValueError(f"Can't calculate purity with shape: {state.shape}")

def fidelity(state1, state2, check=1):
    """ Calculate the fidelity between two quantum states. """
    state1 = as_state(state1, check=check)
    state2 = as_state(state2, check=check)

    if len(state1.shape) == 1 and len(state2.shape) == 1:
        return np.abs(state1 @ state2.conj())**2
    elif len(state1.shape) == 2 and len(state2.shape) == 1:
        return np.abs(state2.conj() @ state1 @ state2)
    elif len(state1.shape) == 1 and len(state2.shape) == 2:
        return np.abs(state1.conj() @ state2 @ state1)
    elif len(state1.shape) == 2 and len(state2.shape) == 2:
        # state1_sqrt = matsqrth_psd(state1)
        # return np.trace(matsqrth(state1_sqrt @ state2 @ state1_sqrt))**2  # textbook formula
        # return np.sum(np.sqrt(eigvals(state1 @ state2)))**2  # this is correct and faster
        state1_sqrt = matsqrth_psd(state1)
        state2_sqrt = matsqrth_psd(state2)
        S = svd(state1_sqrt @ state2_sqrt, compute_uv=False) # this is in between in efficiency, but the more stable
        return np.sum(S)**2
    else:
        raise ValueError(f"Can't calculate fidelity between {state1.shape} and {state2.shape}")

def trace_distance(rho1, rho2, check=1):
    """Calculate the trace distance between two density matrices."""
    # convert to density matrices if necessary
    rho1, rho2 = dm(rho1, check=check), dm(rho2, check=check)
    return 0.5 * trace_norm(rho1 - rho2)

def schmidt_decomposition(state, subsystem_qubits, coeffs_only=False, filter_eps=1e-10, check=2):
    """Calculate the Schmidt decomposition of a pure state with respect to the given subsystem."""
    state = as_state(state, check=check)
    n = count_qubits(state)

    assert len(state.shape) == 1, f"State must be a vector, but has shape: {state.shape}"
    assert len(subsystem_qubits) <= n-1, f"Too many subsystem qubits: {len(subsystem_qubits)} >= {n}"

    # reorder the qubits so that the subsystem qubits are at the beginning
    subsystem_qubits = list(subsystem_qubits)
    other_qubits = [i for i in range(n) if i not in subsystem_qubits]
    # but only if the subsystem qubits are not already at the beginning
    if subsystem_qubits != list(range(len(subsystem_qubits))):
        state = state.reshape([2]*n).transpose(subsystem_qubits + other_qubits)
    a_jk = state.reshape([2**len(subsystem_qubits), 2**len(other_qubits)])

    def filterS(S):
        S = S[S > filter_eps]
        if check >= 1:
            assert np.isclose(np.sum(S**2), 1), f"Schmidt coefficients are not normalized: {np.sum(S**2)} {S}"
        return S

    # calculate the Schmidt coefficients and basis using SVD
    if coeffs_only:
        S = np.linalg.svd(a_jk, compute_uv=False)
        return filterS(S)
    U, S, V = np.linalg.svd(a_jk, full_matrices=False)
    S = filterS(S)
    U = U[:, :len(S)]
    V = V[:len(S), :]
    return S, U.T, V

def correlation_quantum(state, obs_A, obs_B, check=2):
    n_A = count_qubits(obs_A)
    n_B = count_qubits(obs_B)
    state = as_state(state, check=check)

    obs_AB = np.kron(obs_A, obs_B)
    rho_A = partial_trace(state, list(range(n_A)), reorder=False)
    rho_B = partial_trace(state, list(range(n_A, n_A + n_B)), reorder=False)
    return ev(obs_AB, state, check=min(1, check)) - ev(obs_A, rho_A, check) * ev(obs_B, rho_B, check)

def is_kraus(operators, n_qubits=None, trace_preserving=True, orthogonal=False, check=3, print_errors=True):
    """
    Check if the given operators form a valid Kraus decomposition.
    Check level 3 only needed for contractivity (`trace_preserving=False`), otherwise level 2 is sufficient.
    """
    return is_from_assert(assert_kraus, print_errors)(operators, n_qubits, trace_preserving, orthogonal, check)

def assert_kraus(operators, n_qubits=None, trace_preserving=True, orthogonal=False, check=3, tol=1e-10):
    """ Check if the given operators form a valid Kraus decomposition. """
    Ks = np.asarray(operators)
    # 1. Check shapes
    assert len(Ks) > 0, f"Need at least one operator, but got: {Ks.shape}"
    if len(Ks.shape) == 2:
        Ks = Ks[None, ...]
    assert len(Ks.shape) == 3 and Ks.shape[1] == Ks.shape[2], f"Kraus operators should be square matrices, but got shape: {Ks.shape[1:]}"
    # 2. Check size matches n_qubits
    if n_qubits is not None:
        if not is_iterable(n_qubits):
            n_qubits = [n_qubits]
        n_K = count_qubits(Ks[0])
        assert n_K in n_qubits, f"Kraus operators need to have {n_qubits} qubits but have {n_K}"
    if check < 2:
        return True  # trace and orthogonality checks are expensive
    # 3. Check trace-preserving / contractive
    res = np.sum([K.conj().T @ K for K in Ks], axis=0)
    if trace_preserving:
        assert allclose0(res - np.eye(Ks.shape[-1]), tol), f"Operators are not trace-preserving"
    else:
        if check >= 3:
            assert np.max(np.abs(eigvalsh(res))) < 1 + tol, f"Operators are not contractive"
    # 4. Check orthogonality
    if orthogonal:
        for K1, K2 in itertools.combinations(Ks, 2):
            res = np.trace(K1.conj().T @ K2)
            assert np.abs(res) < 1e-10, f"Operators are not orthogonal: {res}"

def measurement_operator(outcome, n, subsystem, as_matrix=True):
    if subsystem is None:
        subsystem = range(n)
    subsystem = list(subsystem)
    q = len(subsystem)
    assert 0 < q <= n, f"Invalid subsystem: {subsystem}"
    assert is_int(outcome) and 0 <= outcome < 2**q, f"Invalid outcome: {outcome}"
    outcome = int(outcome)
    Pi = np.zeros((2**q, 2**(n-q)), dtype=complex)  # just the diagonal
    Pi[outcome] = 1
    Pi_order = subsystem + [i for i in range(n) if i not in subsystem]
    Pi_order_inv = [Pi_order.index(i) for i in range(n)]
    Pi = Pi.reshape([2]*n).transpose(Pi_order_inv).reshape(-1)

    # Pi = np.zeros(2**n, dtype=complex)
    # outcome_bits = format(outcome, f'0{q}b')
    # full_bits = ['0']*n
    # for j, pos in enumerate(subsystem):
    #     full_bits[pos] = outcome_bits[j]
    # for i in range(2**(n-q)):
    #     i_bits = format(i, f'0{n-q}b')
    #     curr_i_bit = 0
    #     for j in range(n):
    #         if j not in subsystem:
    #             full_bits[j] = i_bits[curr_i_bit]
    #             curr_i_bit += 1
    #     idx = int(''.join(full_bits), 2)
    #     Pi[idx] = 1

    if as_matrix:
        return np.diag(Pi)
    return Pi

def POVM(n, subsystem=None, as_matrix=True):
    """
    Create the POVM operators for a projective measurement on the given subsystem in the standard basis.
    They form an orthogonal set of orthogonal projectors, as well as a valid quantum channel (Kraus decomposition).
    """
    if subsystem is None:
        subsystem = range(n)
    return [measurement_operator(outcome, n, subsystem, as_matrix) for outcome in range(2**len(subsystem))]

def reset_operator(n, value=0):
    """
    Create a set of Kraus operators that reset `n` qubits to `value` in the standard basis.
    """
    return [op(value, i, n) for i in range(2**n)]

def choi_from_operators(operators, n=None, check=3):
    """
    Create the Choi matrix from a set of Kraus operators.
    """
    operators = np.asarray(operators)
    n = n or count_qubits(operators[0])
    assert_kraus(operators, n_qubits=n, check=check)
    Choi = np.zeros((2**(2*n), 2**(2*n)), dtype=complex)
    for K in operators:
        Kvec = K.reshape(-1, 1)  # column vectorization
        Choi += Kvec @ Kvec.conj().T
    return Choi

def operators_from_choi(choi, n=None, filter_eps=1e-10, check=3):
    """
    Create the Kraus operators from a Choi matrix.
    """
    choi = np.asarray(choi)
    n = n or int(np.log2(choi.shape[0]) / 2)
    assert choi.shape == (2**(2*n), 2**(2*n)), f"Choi matrix has invalid shape: {choi.shape} ≠ {(2**(2*n), 2**(2*n))}"
    U, S, _ = svd(choi)
    mask = S > filter_eps
    S = S[mask]
    U = U[:, mask]
    S = np.sqrt(S)
    operators = [S[i] * U[:, i].reshape(2**n, 2**n) for i in range(len(S))]
    assert_kraus(operators, n_qubits=n, check=check)
    return operators