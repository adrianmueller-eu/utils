import sys
import numpy as np
from math import prod, log2, ceil
import itertools
try:
    import scipy.sparse as sp
except ImportError:
    pass

from .utils import count_qubits, partial_trace, reorder_qubits, verify_subsystem
from .state import op, ket, dm, ev, as_state, ensemble_from_state, assert_state
from ..mathlib import trace_norm, matsqrth_psd, allclose0, is_square, eigvalsh, svd, is_eye, trace_product, random_isometry
from ..prob import entropy, check_probability_distribution
from ..utils import is_from_assert, is_int, warn

def von_neumann_entropy(state, check=2):
    """ Calculate the von Neumann entropy of a given density matrix. """
    state = as_state(state, check=check)
    if len(state.shape) == 1:
        return 0  # pure state
    # if it just has a single 1, it's also a pure state
    # if np.count_nonzero(state) == 1:
    #     return 0
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

def purity(state, check=1):
    """ Calculate the purity of a quantum state. """
    state = as_state(state, check=check)
    if state.ndim == 1:
        return 1  # ket
        # return np.abs(state @ state.conj())
    elif state.ndim == 2:
        return trace_product(state, state).real
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

def schmidt_decomposition(state, subsystem, coeffs_only=False, filter_eps=1e-10, check=1):
    """Calculate the Schmidt decomposition of a pure state with respect to the given subsystem."""
    state = ket(state, renormalize=check>0, check=check)
    state = reorder_qubits(state, subsystem, separate=True)
    assert state.ndim == 2, f"Subsystem needs to be a bipartition, but was: {subsystem} {state.shape}"

    def filterS(S):
        S = S[S > filter_eps]
        if check >= 1:
            assert np.isclose(np.sum(S**2), 1), f"Schmidt coefficients are not normalized: {np.sum(S**2)} {S}"
        return S

    # calculate the Schmidt coefficients and basis using SVD
    if coeffs_only:
        S = np.linalg.svd(state, compute_uv=False)
        return filterS(S)
    U, S, V = np.linalg.svd(state, full_matrices=False)
    S = filterS(S)
    U = U[:, :len(S)]
    V = V[:len(S), :]
    return S, U.T, V

def schmidt_rank(state, subsystem, tol=1e-10, check=1):
    """Calculate the Schmidt rank of a pure state with respect to the given subsystem."""
    state = ket(state, renormalize=check>0, check=check)
    # if is_pure_dm(partial_trace(state, subsystem)):
    #     return 1
    state = reorder_qubits(state, subsystem, separate=True)
    assert state.ndim == 4, f"Subsystem needs to be a bipartition, but was: {subsystem} {state.shape}"
    return np.linalg.matrix_rank(state, tol=tol)

def schmidt_operator_rank(op, subsystem, tol=1e-10):
    """Calculate the Schmidt operator rank of a given operator with respect to the given subsystem."""
    op = reorder_qubits(op, subsystem, separate=True)
    if op.ndim == 2:  # q == 0 or q == n
        return 1
    op = op.transpose(0, 2, 1, 3).reshape(2**(2*len(subsystem)), -1)
    return np.linalg.matrix_rank(op, tol=tol)

def purify(rho, min_rank=1, ancilla_basis_cb=None, filter_eps=1e-10, check=3):
    probs, kets = ensemble_from_state(rho, filter_eps=filter_eps, check=check)
    r = max(len(probs), min_rank)
    n_ancillas = ceil(log2(r))
    if n_ancillas == 0:
        return kets[0]
    pkets = np.sqrt(probs)[:, None] * kets

    if ancilla_basis_cb is None:
        # ancilla_basis = np.eye(2**n_ancillas, dtype=complex)
        state = np.zeros((2**n_ancillas, len(kets[0])), dtype=complex)
        state[:len(kets)] = pkets
        return state.T.ravel()
    # general basis
    ancilla_basis = ancilla_basis_cb(n_ancillas)
    return np.tensordot(pkets, ancilla_basis[:len(probs)], axes=(0, 0)).reshape(-1)

def stinespring_dilation(ops, min_rank=1, ancilla_basis_cb=None, check=3):
    ops = assert_kraus(ops, allow_reshaped=False, check=check)
    r = max(len(ops), min_rank)
    n_ancilla = ceil(log2(r))
    if ancilla_basis_cb is None:
        dout, din = ops[0].shape
        V = np.zeros((dout*2**n_ancilla, din), dtype=complex)
        for i, K in enumerate(ops):
            V[i*dout:(i+1)*dout, :] = K
        return V    # V is an isometry
    # general basis
    ancilla_basis = ancilla_basis_cb(n_ancilla)
    return sum(np.kron(K, a) for K, a in zip(ops, ancilla_basis))

def correlation_quantum(state, obs_A, obs_B, check=2):
    n_A = count_qubits(obs_A)
    n_B = count_qubits(obs_B)
    state = as_state(state, check=check)

    obs_AB = np.kron(obs_A, obs_B)
    rho_A = partial_trace(state, list(range(n_A)), reorder=False)
    rho_B = partial_trace(state, list(range(n_A, n_A + n_B)), reorder=False)
    return ev(obs_AB, state, check=min(1, check)) - ev(obs_A, rho_A, check) * ev(obs_B, rho_B, check)

def get_channel_dims(operators, as_qubits=False):
    if as_qubits:
        dim_out, dim_in = get_channel_dims(operators, as_qubits=False)
        return count_qubits(dim_out), count_qubits(dim_in)

    K = operators[0]
    if not hasattr(K, 'shape'):
        K = np.asarray(K)
    if K.ndim == 2:
        return K.shape
    elif K.ndim == 3:
        return prod(K.shape[:2]), K.shape[2]
    raise ValueError(f"Not a valid shape for an operator: {K.shape}")

def is_kraus(operators, n=(None, None), allow_reshaped=False, trace_preserving=True, orthogonal=False, check=3, tol=1e-10, print_errors=True):
    """ Check if the given operators form a valid Kraus decomposition. See `assert_kraus` for detailed doc. """
    return is_from_assert(assert_kraus, print_errors)(operators, n, allow_reshaped, trace_preserving, orthogonal, check, tol)

def assert_kraus(operators, n=(None, None), allow_reshaped=False, trace_preserving=True, orthogonal=False, check=3, tol=1e-10):
    """
    Ensures the given `operators` form a valid representation of a quantum channel (Kraus operators). Throws an AssertionError if otherwise.

    Parameters:
        operators (np.ndarray | list[np.ndarray]): List of Kraus operators to check
        n_qubits (tuple): Tuple of expected (n_qubits_out, n_qubits_in)
        trace_preserving (bool): Whether to check for trace-preserving $\\sum K_i^\\dagger K_i = I$ or contractive $\\sum K_i^\\dagger K_i \\leq I$ property
        orthogonal (bool): Check if the operators are orthogonal $\\sum K_i^\\dagger K_j = 0$ for $i \\neq j$
        check (int): Check level (0: ndim only, 1: check with n_qubits argument, 2: trace-preserving + orthogonality, 3: contractivity)
        tol (float): Tolerance (for trace-preserving/contractive property and orthogonality)

    Returns:
        np.ndarray: Kraus operators as a numpy array of shape (n_ops, n_out, n_in).
    """
    # 1. Check ndim
    assert len(operators) > 0, "No operators provided"
    if isinstance(operators, list) and not isinstance(operators[0], np.ndarray):
        np.asarray(operators)  # check same shape of all operators
    if isinstance(operators, np.ndarray):
        if operators.ndim == 2:
            operators = [operators]
        elif allow_reshaped:
            # just based on ndim, we can't distinguish list of 2d operators vs a single 3d operator
            # -> assume it's a list
            # if 4d it's definitely a list of 3d operators
            operators = list(operators)
    K = operators[0]
    if allow_reshaped:
        assert K.ndim in (2,3), f"Operators must be a list of 2D or 3D arrays, but got {K.shape}"
    else:
        assert K.ndim == 2, f"Operators must be a list of 2D arrays, but got {K.shape}"

    if check < 1:
        return operators

    # 2. Check size matches n parameter
    n_out, n_in = get_channel_dims(operators, as_qubits=True)
    if is_int(n):
        n = (n, n)
    if n[0] is not None:
        assert n[0] == n_out, f"Operators have invalid shape for {n[0]} output qubits: {2**n[0]} x {K.shape}"
    if n[1] is not None:
        assert n[1] == n_in, f"Operators have invalid shape for {n[1]} input qubits: {K.shape} x {2**n[1]}"

    if check < 2:
        return operators  # trace and orthogonality checks are expensive
    operators_reshaped = [op.reshape(2**n_out, 2**n_in) for op in operators]

    # 3. Check trace-preserving / contractive
    res = np.sum([K.conj().T @ K for K in operators_reshaped], axis=0)
    if trace_preserving:
        assert is_eye(res, tol), f"Operators are not trace-preserving:\n{res}"
    elif check >= 3:
        evals = eigvalsh(res)
        assert np.max(np.abs(evals)) < 1 + tol, f"Operators are not contractive: {evals}"

    # 4. Check orthogonality
    if orthogonal:
        for i, Ki in enumerate(operators_reshaped):
            for j, Kj in enumerate(operators_reshaped):
                if i == j:
                    continue
                res = np.trace(Ki.conj().T @ Kj)
                assert np.abs(res) < tol, f"Operators {i,j} are not orthogonal: {res}"
    return operators

def is_isometric_channel(operators, check=3):
    """ Check if given operators form an isometric quantum channel. """
    return len(operators) == 1 and (check == 0 or is_kraus(operators, check=check))

def is_square_channel(operators, check=3):
    try:
        operators = assert_kraus(operators, allow_reshaped=True, check=check)
    except AssertionError:
        return False
    o = operators[0]
    return (o.ndim == 2 and o.shape[0] == o.shape[1]) or (o.ndim == 3 and prod(o.shape[:2]) == o.shape[2])

def is_unitary_channel(operators, check=3):
    """ Check if given operators form a unitary quantum channel. """
    return is_isometric_channel(operators, check=check) and is_square_channel(operators, check=0)

def apply_channel(operators, state, reshaped, check=3):
    # sanity checks
    if check:
        state = np.asarray(state)
        assert state.ndim in (1, 2, 4), f"Invalid state shape: {state.shape}"
        # convert to reshaped=False to check state
        if not reshaped:
            tmp_state = state
        elif state.ndim == 2:
            tmp_state = state.ravel()
        else:
            tmp_state = state.reshape(prod(state.shape[:2]), -1)
        n = count_qubits(tmp_state)
        assert_state(tmp_state, n=n, check=check)
        operators = assert_kraus(operators, n=(None, n), allow_reshaped=False, check=check)
        assert operators[0].shape[1] == state.shape[0], f"Input dimension of the operators does not match the state dimension: {operators[0].shape} x {state.shape}"

    state_is_dm = reshaped and state.ndim == 4 or not reshaped and state.ndim == 2
    if state_is_dm:
        n_out = operators[0].shape[0]
        if reshaped:
            assert state.shape[1] == state.shape[3], f"State must be a square matrix: {state.shape}"
            new_state = np.zeros((n_out, state.shape[1], n_out, state.shape[1]), dtype=complex)
        else:
            new_state = np.zeros((n_out, n_out), dtype=complex)
        for K in operators:
            if not reshaped:
                # (m x q) x (q x q) x (q x m) -> m x m
                new_state += K @ state @ K.T.conj()
            else:
                # (m x q) x (q x (n-q) x q x (n-q)) -> m x (n-q) x q x (n-q)
                tmp = np.tensordot(K, state, axes=1)
                # (m x (n-q) x q x (n-q)) x (q x m) -> m x (n-q) x (n-q) x m
                tmp = np.tensordot(tmp, K.T.conj(), axes=(2,0))
                # m x (n-q) x (n-q) x m -> m x (n-q) x m x (n-q)
                new_state += tmp.transpose([0, 1, 3, 2])  # TODO: change convention to q x (n-q) x (n-q) x q to avoid the transpose here? (5~10% faster)
    else:
        if check:
            assert is_unitary_channel(operators, check=0), "Non-unitary operators can't be applied to state vectors!"
        U = operators[0]
        # (q x q) x (q x (n-q)) -> q x (n-q)  or  (q x q) x q -> q
        new_state = np.tensordot(U, state, axes=1)
    return new_state

def combine_channels(operators1, operators2, filter0=True, tol=1e-10, check=3):
    """
    Combine two quantum channels to a single quantum channel in the order $E = E_1 \\circ E_2$.
    `filter0` removes zero operators from the result with tolerance `tol`.
    """
    operators1 = assert_kraus(operators1, allow_reshaped=False, check=check)
    operators2 = assert_kraus(operators2, allow_reshaped=True, check=check)
    assert operators1[0].shape[1] == operators2[0].shape[0], f"Input dimension of `operators1` does not match output dimension of `operators2`: {operators1[0].shape} x {operators2[0].shape}"

    new_operators = []
    for Ki in operators1:
        for Kj in operators2:
            # (m x q) x (q x (n-q) x -1) -> m x (n-q) x -1  or  (m x q) x (q x -1) -> m x -1
            Kij = np.tensordot(Ki, Kj, axes=1)
            if not filter0 or not allclose0(Kij, tol):
                new_operators.append(Kij)
    # sanity check
    assert len(new_operators) > 0, f"Combined channel is empty. Filter tolerance too large ({tol:.6g})?"
    return new_operators

def update_choi(operators, choi, sparse=True, check=3):
    # if self._operators.ndim == 2:  # (out*in) x (out*in)
    #     choi = choi_from_channel(operators, check=0)
    #     return choi @ self._operators
    operators = assert_kraus(operators, allow_reshaped=False, check=check)
    assert choi.ndim in (4,6)  # n x n_in x n x n_in  or  q x (n-q) x n_in x q x (n-q) x n_in
    d_out, d_in = operators[0].shape[0], choi.shape[-1]
    if choi.ndim == 4:
        shape = (d_out, d_in, d_out, d_in)
    else:
        d_nq = choi.shape[1]
        shape = (d_out, d_nq, d_in, d_out, d_nq, d_in)
    if sparse:
        Ks = [sp.coo_array(o) for o in operators]
        new_choi = sp.coo_array(shape, dtype=choi.dtype)
    else:
        Ks = [np.asarray(o) for o in operators]
        new_choi = np.zeros(shape, dtype=choi.dtype)
    axes = ([choi.ndim//2], [0])

    for K in Ks:
        if sparse:
            # (m x q) x (q x (n-q) x n_in x q x (n-q) x n_in) -> m x (n-q) x n_in x q x (n-q) x n_in
            # or  (m x q) x (q x n_in x q x n_in) -> m x n_in x q x n_in
            tmp = K.tensordot(choi, axes=1)
            # (m x (n-q) x n_in x q x (n-q) x n_in) x (q x m) -> m x (n-q) x n_in x m x (n-q) x n_in
            # or  (m x n_in x q x n_in) x (q x m) -> m x n_in x m x n_in
            tmp = tmp.tensordot(K.conj().T, axes=axes)
        else:
            tmp = np.tensordot(K, choi, axes=1)
            tmp = np.tensordot(tmp, K.conj().T, axes=axes)

        if choi.ndim == 4:  # TODO: Similarly here, change the convention to have q at the outsides: q x (n-q) x n_in x n_in x (n-q) x q. Then remove the transpose.
            tmp = tmp.transpose([0, 1, 3, 2])
        else:
            tmp = tmp.transpose([0, 1, 2, 5, 3, 4])
        new_choi += tmp

    if sparse:
        # actually execute the addition
        s = new_choi.shape
        new_choi = sp.csr_array(new_choi.reshape(-1)).reshape(s)
    return new_choi

def measurement_operator(outcome, n, subsystem=None, as_matrix=True):
    if subsystem is None:
        subsystem = range(n)
    subsystem = verify_subsystem(subsystem, n)
    q = len(subsystem)
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

def reset_channel(new_state=0, n=None, filter_eps=1e-10, check=3):
    """
    Create a set of Kraus operators that reset `n` qubits to `value` in the standard basis.
    """
    new_state = as_state(new_state, renormalize=False, n=n, check=check)
    p, kets = ensemble_from_state(new_state, filter_eps=filter_eps, check=check)
    sp = np.sqrt(p)
    n = count_qubits(kets[0])
    Ks = []
    for sp_i, k_i in zip(sp, kets):
        for z in range(2**n):
            K = sp_i * op(k_i, z, n=n, check=check)
            Ks.append(K)
    return Ks

def extension_channel(new_state, n=None, filter_eps=1e-10, check=3):
    """
    Create a set of Kraus operators that expand to `n` new qubits initialized in `new_state`.
    """
    new_state = as_state(new_state, renormalize=False, n=n, check=check)
    p, kets = ensemble_from_state(new_state, filter_eps=filter_eps, check=check)
    sp = np.sqrt(p)
    n = count_qubits(kets[0])
    Ks = []
    for sp_i, k_i in zip(sp, kets):
        K = sp_i * k_i[:, None]  # expand to n qubits
        Ks.append(K)
    return Ks

def removal_channel(n):
    """
    Create a set of Kraus operators that remove `n` qubits. This is equivalent to the (partial) trace operation.
    """
    return list(np.eye(2**n)[:,None,:])  # [ket(i, n=q)[None,:] for i in range(2**q)]

def random_channel(n_out, n_in=None):
    """
    Sample a random quantum channel.
    """
    if n_in is None:
        n_in = n_out
    assert is_int(n_out) and n_out >= 0, f"Invalid number of output qubits: {n_out}"
    assert is_int(n_in) and n_in >= 0, f"Invalid number of input qubits: {n_in}"

    N = n_out + n_in
    V = random_isometry(2**N, 2**n_in)
    rem = removal_channel(N - n_out)
    V = V.reshape(2**(N - n_out), 2**n_out, 2**n_in)
    Ks = combine_channels(rem, [V])
    return [K[0] for K in Ks]  # remove inital axis of shape (1,)

def average_channel(channels, p=None, check=3):
    """
    Average the channels into a single channel. If p is None, assume uniform weights.
    """
    if check:
        channels = [assert_kraus(ops, allow_reshaped=True, check=check) for ops in channels]
        # check they all have the same shape
        for i, ops in enumerate(channels):
            assert ops[0].shape == channels[0][0].shape, f"Operators {i} have inconsistent shape: {ops[0].shape} != {channels[0][0].shape}"

    C = len(channels)
    if p is None:
        p = np.ones(C) / C
    else:
        p = check_probability_distribution(p, check=check)
        assert len(p) == C, "The length of p must match the number of channels"

    # perform averaging
    p = np.sqrt(p)
    ops = []
    for i, ops_i in enumerate(channels):
        for o in ops_i:
            ops.append(p[i] * o)
    return ops

def choi_from_channel(operators, sparse='auto', filter_eps=sys.float_info.epsilon, check=3):
    """
    Create the Choi matrix from a set of Kraus operators.
    """
    operators = assert_kraus(operators, allow_reshaped=False, check=check)
    choi_dim = prod(operators[0].shape)  # 2**(2*n) if input space == output space

    if sparse == 'auto':
        if choi_dim <= 64:  # n <= 3 qubits use dense
            sparse = False
        else:
            thresh = 0.25
            # heuristic for percentage of non-zero elements in the choi matrix
            fill = np.mean([(np.count_nonzero(K)/K.size)**2 for K in operators])
            sparse = fill < thresh

    if sparse:
        choi = sp.csr_array((choi_dim, choi_dim), dtype=complex)
        operators = [sp.csr_array(K) for K in operators]
        # # filter small entries from operators  # TODO: check performance
        # for K in operators:
        #     K.data[np.abs(K.data) < filter_eps] = 0
        #     K.eliminate_zeros()
    else:
        choi = np.zeros((choi_dim, choi_dim), dtype=complex)
        if sp.issparse(operators[0]):
            operators = [K.toarray() for K in operators]

    for K in operators:
        Kvec = K.reshape(-1, 1)
        choi += Kvec @ Kvec.conj().T  # outer product

    # # filter small entries  # TODO: check performance
    # if filter_eps > 0 and sparse:
    #     choi.data[np.abs(choi.data) < filter_eps] = 0
    #     choi.eliminate_zeros()

    return choi

def channel_from_choi(choi, dims=(None, None), filter_eps=1e-12, k=None):
    """
    Create the Kraus operators from a Choi matrix. Either `d_out` or `d_in` must be provided.
    """
    if not hasattr(choi, 'shape'):
        choi = np.asarray(choi)
    assert is_square(choi), f"Choi matrix is not square: {choi.shape}"
    assert choi.ndim == 2, f"Choi matrix must be 2D: {choi.shape}"

    # infer Choi dimensions
    d_out, d_in = dims if not isinstance(dims, int) else (dims, dims)
    assert d_out is not None or d_in is not None, "Either d_out or d_in must be provided"
    choi_dim = choi.shape[0]
    if d_out is None:
        d_out = choi_dim // d_in
        assert choi_dim == d_out*d_in, f"Invalid d_in: {d_in} for {choi.shape}"
    elif d_in is None:
        d_in = choi_dim // d_out
        assert choi_dim == d_out*d_in, f"Invalid d_out: {d_out} for {choi.shape}"
    else:
        choi_dim = d_out*d_in
        assert choi.shape == (choi_dim, choi_dim), f"Choi matrix has invalid shape: {choi.shape} ≠ {(choi_dim, choi_dim)}"

    # warning to the user
    k = k or choi_dim
    k = min(k, choi_dim)
    if choi_dim*k > 1e7:
        warn(f"SVD for {choi.shape} Choi matrix may take a while (k = {k}).")

    # perform SVD
    try:
        assert k < choi_dim//4  # use dense SVD for large k
        U, S, _ = sp.linalg.svds(sp.csr_array(choi), k=k, return_singular_vectors='u')
    except Exception:
        if hasattr(choi, 'toarray'):
            choi = choi.toarray()
        U, S, _ = svd(choi)

    # filter small singular values out
    mask = S > filter_eps
    S = S[mask]
    U = U[:, mask]

    # generate minimal set of Kraus operators
    S = np.sqrt(S)
    operators = [S[i] * U[:, i].reshape(d_out, d_in) for i in range(len(S))]
    # assert_kraus(operators, n_qubits=(n_out, n_in), check=3)
    return operators

def compress_channel(operators, filter_eps=1e-12, check=3):
    """
    Find a minimal set of Kraus operators that represent the same quantum channel.
    """
    operators = assert_kraus(operators, allow_reshaped=True, check=check)

    # reshape operators to 2D
    orig_shape = operators[0].shape  # store the original shape
    d_out, d_in = get_channel_dims(operators)
    ops = [op.reshape(d_out, d_in) for op in operators]

    # obtain choi matrix
    choi = choi_from_channel(ops, filter_eps=filter_eps, check=0)
    # perform SVD to convert back to Kraus operators
    ops_compressed = channel_from_choi(choi, (d_out, d_in), filter_eps=filter_eps, k=len(ops))

    # reshape back to original shape
    ops_compressed = [op.reshape(orig_shape) for op in ops_compressed]
    return ops_compressed

def kron_with_id_channel(choi: np.ndarray, d: int, dims: tuple[int], back=False) -> np.ndarray:
    assert isinstance(choi, np.ndarray), f"Choi matrix must be a numpy array: {type(choi)}"
    assert choi.ndim in (2,4), f"Choi matrix must be 2D or 4D: {choi.shape}"
    was_4d = choi.ndim == 4
    d_out, d_in = dims

    choi = choi.reshape(d_out, d_in, d_out, d_in)
    idcs = itertools.product(range(d), repeat=2)
    if back:
        res = np.zeros((d_out, d, d_in, d)*2, dtype=choi.dtype)
        for i, j in idcs:
            res[:,i,:,i,:,j,:,j] = choi
    else:
        res = np.zeros((d, d_out, d, d_in)*2, dtype=choi.dtype)
        for i, j in idcs:
            res[i,:,i,:,j,:,j,:] = choi

    if was_4d:
        return res.reshape([d_out*d, d_in*d]*2)
    return res.reshape([d_out*d*d_in*d]*2)
