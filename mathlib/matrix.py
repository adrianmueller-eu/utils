import warnings, sys
import numpy as np
import itertools
from functools import reduce
from math import log2, sin, cos, sqrt
import scipy.sparse as sp
from scipy.linalg import eig, eigh, eigvals, eigvalsh, svd, det, inv, pinv
from scipy.linalg import expm as matexp
from scipy.linalg import logm as _matlog
# for use when importing utils
from scipy.linalg import fractional_matrix_power as matpow
from scipy.linalg import sqrtm as matsqrt

from .basic import series, sequence
from ..models import Polynomial

sage_loaded = False
try:
    from sage.all import *
    sage_loaded = True
except ImportError:
    pass

#######################
### Property checks ###
#######################

def _sq_matrix_allclose(a, f, tol=1e-12):
    a = np.asarray(a)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        return False
    # a[np.isnan(a)] = 0
    a, b = f(a)
    return allclose0(a-b, tol)

def is_symmetric(a, tol=1e-12):
    return _sq_matrix_allclose(a, lambda a: (
        a, a.T
    ), tol)

def is_hermitian(a, tol=1e-12):
    return _sq_matrix_allclose(a, lambda a: (
        a, a.conj().T
    ), tol)

def is_antihermitian(a, tol=1e-12):
    return _sq_matrix_allclose(a, lambda a: (
        a, -a.conj().T
    ), tol)

def is_orthogonal(a, tol=1e-12):
    return _sq_matrix_allclose(a, lambda a: (
        a @ a.T, np.eye(a.shape[0])
    ), tol)

def is_unitary(a, tol=1e-12):
    return _sq_matrix_allclose(a, lambda a: (
        a @ a.conj().T, np.eye(a.shape[0])
    ), tol)

def is_involutory(a, tol=1e-12):
    return _sq_matrix_allclose(a, lambda a: (
        a @ a, np.eye(a.shape[0])
    ), tol)

def is_complex(a):
    if hasattr(a, 'dtype'):
        if a.dtype == complex:
            return True
        return np.issubdtype(a.dtype, complex)
    return np.iscomplex(a).any()

def is_psd(a, eigs=None, check=3, tol=1e-12):
    if check >= 2 and not is_hermitian(a, tol):
        return False
    if eigs is None:
        if check >= 3:
            eigs = eigvalsh(a)
            return np.all(eigs >= -tol)
        return True
    # tol = len(eigs)*sys.float_info.epsilon
    return np.all(eigs.real >= -tol) and allclose0(eigs.imag, tol)

def is_normal(a, tol=1e-12):
    return _sq_matrix_allclose(a, lambda a: (
        a @ a.conj().T, a.conj().T @ a
    ), tol)

def is_projection(a, tol=1e-9):
    a = np.asarray(a)
    return _sq_matrix_allclose(a, lambda a: (
        a @ a, a
    ), a.shape[0]*tol)

def is_projection_orthogonal(a, tol=1e-9):
    return is_projection(a, tol=tol) and is_hermitian(a, tol)

def is_diag(a, tol=1e-12):
    # return _sq_matrix_allclose(a, lambda a: (
    #     np.diag(np.diag(a)), a
    # ), rtol=rtol, atol=atol)
    if abs(a[0,-1]) > tol:  # shortcut
        return False

    # a[np.isnan(a)] = 0
    a = a.reshape(-1)[:-1].reshape(a.shape[0]-1, a.shape[1]+1)[:,1:]  # remove diagonal
    return allclose0(a, tol=tol)

def allclose0(a, tol=1e-12):
    if tol == 0:
        return np.all(a == 0)
    a = np.asarray(a)
    if is_complex(a):
        return np.all(np.abs(a.real) <= tol) and np.all(np.abs(a.imag) <= tol)
    return np.all(np.abs(a) <= tol)

########################
### Matrix functions ###
########################

def matfunc(A, f, not_hermitian=False):
    if not not_hermitian and is_hermitian(A):  # is_hermitian takes a small fraction of the timing difference between eig and eigh
        return matfunch(A, f)
    D, T = eig(A)
    D = np.asarray(f(D))
    if not is_complex(T) and allclose0(D.imag):
        D = D.real
    return T @ (D[:,None] * inv(T))

def matlog(A, base=np.e):
    return _matlog(A) / np.log(base)

# def matexp(A, not_h=False):
#     return matfunc(A, np.exp, not_h)

def matexp_series(A):
    return np.eye(A.shape[0]) + series(lambda n, A_pow: A_pow @ A / n, start_value=A, start_index=1)

# def matlog(A, base=np.e, not_h=False):
#     return matfunc(A, lambda x: np.log(x) / np.log(base), not_h)

# def matpow(A, n, not_h=False):
#     return matfunc(A, lambda x: x**n, not_h)

# def matsqrt(A, not_h=False):
#     return matfunc(A, np.sqrt, not_h)

def matfunch(A, f):
    D, U = eigh(A)
    D = np.asarray(f(D.astype(complex)))
    return U @ (D[:,None] * U.conj().T)

def matexph(A):
    return matfunch(A, np.exp)

def matlogh(A, base=np.e):
    return matfunch(A, lambda x: np.log(x) / np.log(base))

def matpowh(A, n):
    return matfunch(A, lambda x: x**n)

def matsqrth(A):
    return matfunch(A, np.sqrt)

def matfunch_psd(A, f):
    # check if eigh is from scipy.linalg or numpy.linalg
    if hasattr(eigh, '__module__') and eigh.__module__.startswith('scipy'):
        # warnings.warn("In scipy, eigh has better performance than svd for PSD matrix functions.", stacklevel=2)
        return matfunch(A, f)
    V, S, Vh = svd(A)
    S = np.asarray(f(S))
    return V @ (S[:,None] * Vh)

def matexph_psd(A):
    return matfunch_psd(A, np.exp)

def matlogh_psd(A, base=np.e):
    return matfunch_psd(A, lambda x: np.log(x) / np.log(base))

def matpowh_psd(A, n):
    return matfunch_psd(A, lambda x: x**n)

def matsqrth_psd(A):
    """ Matrix square root for PSD matrices using SVD """
    return matfunch_psd(A, np.sqrt)

def sinv(A, likely_singular=False, tol=1e-12):
    """
    Matrix inverse.
    """
    A = np.asarray(A)
    if not likely_singular:
        try:
            A_inv = inv(A)
            diff = A @ A_inv - np.eye(A.shape[0])
            if allclose0(diff, tol=tol):
                # detA = det(A)
                # if not np.isnan(detA) and np.abs(detA) < 1e-3:
                #     print(detA)
                return A_inv  # success!
            print(f"inv failed! {np.max(np.abs(diff))}")
        except:
            print("inv failed!")
            # not square or singular
            pass
    return pinv(A)

def normalize(a, p=2, axis=0):
    """
    Normalize a vector (or tensor of vectors). For np.ndarray, operates *inplace* and returns the same object.
    """
    if is_complex(a):
        a = np.asarray(a, dtype=complex)
    else:
        a = np.asarray(a, dtype=float)
    if a.shape == ():
        return a/np.linalg.norm(a)
    a /= np.linalg.norm(a, ord=p, axis=axis, keepdims=True)
    return a

def permutation_sign(p, base):
    if base == 1:
        return 1
    inversions = sum(i > j for i,j in itertools.combinations(p, 2))
    return base**inversions

def immanant(A, char):
    """ Thanks for the cookies! """
    r = range(A.shape[0])
    return sum(permutation_sign(P, char) * np.prod(A[r, P]) for P in itertools.permutations(r))

def permanent(A):
    return immanant(A, 1)

def determinant(A):
    if A.shape[0] >= 10:
        raise ValueError("Please use a proper method like np.linalg.det.")
    # np.prod(eigvals(A))   # O(n^3)
    return immanant(A, -1)  # O(n!)

def characteristic_polynomial(A):
    D = eigvals(A)
    p = Polynomial.from_roots(D)
    return p

def commutator(A, B):
    return A @ B - B @ A

def commute(A, B, tol=1e-10):
    """ Check if two matrices commute. """
    return allclose0(commutator(A, B), tol=tol)

def anticommutator(A, B):
    return A @ B + B @ A

def anticommute(A, B, tol=1e-10):
    """ Check if two matrices anticommute. """
    return allclose0(anticommutator(A, B), tol=tol)

def trace_product(A, B):
    """Hilbert-Schmidt product or trace inner product of two matrices."""
    return np.trace(A.T.conj() @ B)

hilbert_schmidt_product = trace_product

def trace_norm(A):
    """Trace norm or nuclear norm of a matrix."""
    return np.linalg.norm(A, ord='nuc')
    # return np.trace(matsqrt(A.T.conj() @ A))

def frobenius_norm(A):
    """Frobenius norm or Hilbert-Schmidt norm of a matrix."""
    return np.linalg.norm(A, ord='fro')
    # return np.sqrt(np.trace(A.T.conj() @ A))

def polar(A, kind='right', hermitian=False, force_svd=False):
    """
    Polar decomposition of *any* matrix into a unitary and a PSD matrix: `A = UJ` ('right') or `A = KU` ('left'),
    where `J = sqrt(A^H A)` and `K = sqrt(A A^H)`. If `A` is invertible, then `J` and `K` are positive definite
    and `U = A J^(-1) = K^(-1) A` is unitary.
    """
    # eigh + 4x@ is faster than svd + 2x@ for invertible square matrices
    A = np.asarray(A)
    if not force_svd and A.shape[-2] == A.shape[-1] and np.nan_to_num(det(A)) > 1e-6:
        def _get(S):
            D, U_S = eigh(S)
            D_sqrt = np.sqrt(D)[:,None]
            assert np.all(D > 0), f"Matrix is not invertible: {D}"
            J = U_S @ (D_sqrt * U_S.conj().T)
            J_inv = U_S @ (1/D_sqrt * U_S.conj().T)
            return J, J_inv

        AH = A if hermitian else A.T.conj()
        if kind == 'right':
            J, J_inv = _get(AH @ A)
            U = A @ J_inv
            return U, J
        elif kind == 'left':
            K, K_inv = _get(A @ AH)
            U = K_inv @ A
            return K, U
        raise ValueError(f"Unknown kind '{kind}'.")

    # see https://en.wikipedia.org/wiki/Polar_decomposition#General_derivation
    W, S, Vh = svd(A, full_matrices=False)
    U = W @ Vh
    # A = W S V^H = U(V S V^H) = (W S W^H)U
    if kind == 'right':
        J = Vh.conj().T @ (S[:,None] * Vh)
        return U, J
    elif kind == 'left':
        K = W @ (S[:,None] * W.conj().T)
        return K, U
    raise ValueError(f"Unknown kind '{kind}'.")

# def svd2(A):
#     S, J = polar(A, kind='right')
#     D, T = eig(J)
#     return S @ T, np.diag(D), T.conj().T

# def cholesky(A, check=3):  # very slow, use scipy.linalg.cholesky or np.linalg.cholesky
#     """ Cholesky decomposition of a PSD matrix into a lower triangular matrix $A = LL^*$ """
#     if is_psd(A, check=check):
#         raise ValueError('Cholesky decomposition works only for PSD matrices!')
#     L = np.zeros_like(A)
#     for i in range(A.shape[0]):
#         L[i, i] = np.sqrt(A[i, i] - L[i, :i] @ L[i, :i])
#         L[i+1:, i] = (A[i+1:, i] - L[i+1:, :i] @ L[i, :i].conj()) / L[i, i]
#     return L

def gram_schmidt(A, normalized=True):
    """ Orthonormalize a set of vectors using the Gram-Schmidt process. Consider using `np.linalg.qr` for better performance. """
    A = np.asarray(A)
    basis = []
    for v in A.conj().T:
        w = v - np.sum(np.vdot(v,b)*b for b in basis)
        if (w > 1e-10).any():
            basis.append(normalize(w) if normalized else w)
    return np.array(basis)

def power_iteration(A, eps=1e-8):
    """ Power iteration algorithm. If A has a real, positive, unique eigenvalue of largest magnitude, this outputs it and the associated eigenvector."""
    eigvec = sequence(
        lambda i, b: normalize(A @ b),
        start_value=random_vec(A.shape[1]),
        eps=eps
    )
    eigval = eigvec.T.conj() @ A @ eigvec
    return eigval, eigvec

#######################
### Hermitian bases ###
#######################

if not sage_loaded:

    def SO(n):
        """ Special orthogonal group. Returns n(n-1)/2 functions that take an angle and return the corresponding real rotation matrix """
        def rotmat(i, j, phi):
            a = np.eye(n)
            cp, sp = cos(phi), sin(phi)
            a[i,i] = cp
            a[j,j] = cp
            a[i,j] = -sp
            a[j,i] = sp
            return a
        return [lambda phi: rotmat(i, j, phi) for i,j in itertools.combinations(range(n), 2)]

    def su(n, include_identity=False, sparse=False, normalize=False):
        """ The Lie algebra associated with the Lie group SU(n). Returns the n^2-1 generators (traceless Hermitian matrices) of the group. Use `include_identity = True` to return a complete orthogonal basis of hermitian `n x n` matrices.

        Parameters
            n (int): The dimension of the matrices.
            include_identity (bool, optional): If True, include the identity matrix in the basis (default: False).
            sparse (bool, optional): If True, return a sparse representation of the matrices (default: False).
            normalize (bool, optional): If True, normalize the matrices to have norm 1 (default: False).

        Returns
            list[ np.ndarray | scipy.sparse.csr_array ]: A list of `n^2-1` matrices that form a basis of the Lie algebra.
        """
        if sparse:
            base = sp.lil_array((n,n), dtype=complex)
        else:
            if n > 100:
                warnings.warn(f"For `n = {n} > 100`, it is recommended to use `sparse=True` to save memory.", stacklevel=2)
            base = np.zeros((n,n), dtype=complex)

        basis = []

        # Identity matrix, optional
        if include_identity:
            identity = base.copy()
            for i in range(n):
                identity[i,i] = 1
            if normalize:
                # factor 2 to get norm sqrt(2), too
                identity = sqrt(2/n) * identity
            basis.append(identity)

        # Generate the off-diagonal matrices
        for i in range(n):
            for j in range(i+1, n):
                m = base.copy()
                m[i,j] = 1
                m[j,i] = 1
                basis.append(m)

                m = base.copy()
                m[i, j] = -1j
                m[j, i] = 1j
                basis.append(m)

        # Generate the diagonal matrices
        for i in range(1,n):
            m = base.copy()
            for j in range(i):
                m[j,j] = 1
            m[i,i] = -i
            if i > 1:
                m = sqrt(2/(i*(i+1))) * m
            basis.append(m)

        if normalize:
            # su have norm sqrt(2) by default
            basis = [m/sqrt(2) for m in basis]
        if sparse:
            # convert to csr format for faster arithmetic operations
            return [sp.csr_matrix(m) for m in basis]
        return basis

    def SU(n):
        """ Special unitary group. Returns n^2-1 functions that take an angle and return the corresponding complex rotation matrix """
        generators = su(n)
        def rotmat(G):
            D, U = eigh(G)
            return lambda phi: U @ (np.exp(-1j*phi/2*D)[:,None] * U.conj().T)
        return [rotmat(G) for G in generators]

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
            # apply norm sqrt(2**n) to the first element, and reduce the rest
            first = l[0]/sqrt(2**n)
            if len(l) == 1:
                return first
            rest = reduce(f, l[1:])
            return f(first, rest)
        else:
            return reduce(f, l)

    if kind == 'str':
        norm_str = f"{1/sqrt(2**n)}*" if normalize else ""
        return [norm_str + ''.join(i) for i in itertools.product(['I', 'X', 'Y', 'Z'], repeat=n)]
    I, X, Y, Z = np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])
    if kind == 'np':
        return [reduce_norm(np.kron, i, normalize) for i in itertools.product([I,X,Y,Z], repeat=n)]
    elif kind == 'sp':
        basis = [sp.csr_array(b) for b in [I,X,Y,Z]]
        return [reduce_norm(sp.kron, i, normalize) for i in itertools.product(basis, repeat=n)]
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
    n = int(log2(H.shape[0]))
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

##############
### Random ###
##############

def random_vec(size, params=(0,1), complex=False, kind='uniform'):
    if kind == 'uniform':
        rng = np.random.uniform
    elif kind == 'normal':
        rng = np.random.normal
    else:
        raise ValueError(f"Unknown kind '{kind}'.")
    if complex:
        return rng(*params, size=size) + 1j*rng(*params, size=size)
    return rng(*params, size=size)

def random_square(size, params=(0,1), complex=False, kind='normal'):
    if not hasattr(size, '__len__'):
        size = (size, size)
    if size[0] != size[1] or len(size) != 2:
        raise ValueError(f"The shape must be square, but was {size}.")
    return random_vec(size, params=params, complex=complex, kind=kind)

def random_symmetric(size, params=(0,1)):
    a = random_square(size, params=params)
    return (a + a.T)/2

def random_orthogonal(size, params=(0,1)):
    a = random_square(size, params=params, complex=False)
    return np.linalg.qr(a)[0]

def random_hermitian(size, params=(0,1)):
    a = random_square(size, params=params, complex=True)
    return (a + a.conj().T)/2

def random_unitary(size, kind='haar'):
    if kind == 'haar':
        A = random_square(size, complex=True, kind='normal')
        Q, R = np.linalg.qr(A)
        R_d = np.diag(R)
        L = R_d / np.abs(R_d)
        return Q * L[None,:]
    elif kind == 'polar':  # fastest for very small and slowest for very large matrices
        A = random_square(size, complex=True, kind='normal')
        D, U = eigh(A.T.conj() @ A)
        D_sqrt = np.sqrt(D)[:,None]
        J_inv = U @ (1/D_sqrt * U.conj().T)
        return A @ J_inv
    elif kind == 'hermitian':
        H = random_hermitian(size)
        return matexp(1j*H)
    else:
        raise ValueError(f"Unknown kind '{kind}'.")

def random_psd(size, params=(0,1), complex=True):
    params = (params[0], sqrt(params[1]))  # eigs scale with variance
    a = random_square(size, params=params, complex=complex, kind='normal')
    return a @ a.conj().T

def random_normal(size, params=(0,1), complex=True):
    U = random_unitary(size)
    D = random_vec(U.shape[0], params=params, complex=complex, kind='normal')
    return U @ (D[:,None] * U.conj().T)

def random_projection(size, rank=None, orthogonal=True, complex=True):
    if rank is None:
        rank = np.random.randint(1, size+orthogonal)  # rank == n is always orthogonal (identity)
    else:
        rank = min(rank, size)

    # if orthogonal:
    #     # P^2 = P and P = P^H
    #     U = random_unitary(size)
    #     D = np.random.permutation([1]*rank + [0]*(size-rank))
    #     return U @ (D[:,None] * U.conj().T)
    # else:
    #     # P^2 = P, but almost never P = P^H
    #     A = random_square(size, complex=True)
    #     D = np.random.permutation([1]*rank + [0]*(size-rank))
    #     return A @ (D[:,None] * inv(A))

    # much faster for rank << size
    A = random_vec((size, rank), complex=complex)
    if orthogonal:
        B = A.conj().T
    else:
        B = random_vec((rank, size), complex=complex)
    return A @ sinv(B @ A, tol=size*1e-9) @ B