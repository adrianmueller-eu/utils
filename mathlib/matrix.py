import warnings, sys
import numpy as np
import itertools
from math import log2, sin, cos, sqrt, factorial, ceil, log10, prod
from functools import reduce
try:
    import scipy
    import scipy.sparse as sp
    from scipy.linalg import eig, eigvals, svd, det, inv, pinv, schur
except ImportError:
    from numpy.linalg import eig, eigvals, svd, det, inv, pinv

from ..utils import shape_it, size_samples

def eigh(A, **kwargs):
    """Eigenvalue decomposition of a hermitian matrix."""
    if len(A) <= 68:
        return np.linalg.eigh(A, **kwargs)
    return scipy.linalg.eigh(A, **kwargs)

def eigvalsh(A, **kwargs):
    """Eigenvalues of a hermitian matrix."""
    if len(A) <= 68:
        return np.linalg.eigvalsh(A, **kwargs)
    return scipy.linalg.eigvalsh(A, **kwargs)

from .basic import series, sequence
from .number_theory import Group, mod_roots
from ..models import Polynomial
from ..utils import is_int, is_iterable

sage_loaded = False
try:
    from sage.all import *
    sage_loaded = True
except ImportError:
    pass

def outer(a, b=None):
    # faster than np.outer for small vectors and batch-enabled
    if b is None:
        b = a.conj()
    return a[...,None] * b[...,None,:]

#######################
### Property checks ###
#######################

def _sq_matrix_allclose(a, f, tol=1e-12):
    a = np.asarray(a)
    assert is_square(a), "Expected square matrix, got {a.shape}"
    # a[np.isnan(a)] = 0
    a, b = f(a)
    # considerable speedup for large matrices if they are both contiguous
    if a.shape[0] > 1000 and a.flags['C_CONTIGUOUS'] != b.flags['C_CONTIGUOUS']:
        if a.flags['F_CONTIGUOUS']:
            a = a.copy()
        else:  # if b.flags['F_CONTIGUOUS']:
            b = b.copy()
    return allclose0(a-b, tol)

def is_symmetric(a, tol=1e-12):
    return _sq_matrix_allclose(a, lambda a: (
        a, a.T
    ), tol)

def is_antisymmetric(a, tol=1e-12):
    return _sq_matrix_allclose(a, lambda a: (
        a, -a.T
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
    return is_square(a) and is_eye(a @ a.T, tol=tol)

def is_unitary(a, tol=1e-12):
    return is_square(a) and is_isometry(a, tol=tol)

def is_isometry(a, kind='right', tol=1e-12):
    a = np.asarray(a)
    if kind == 'right':
        return is_eye(a.conj().T @ a, tol=tol)
    elif kind == 'left':
        return is_eye(a @ a.conj().T, tol=tol)
    raise ValueError(f"Unknown kind '{kind}'. Use 'right' or 'left'.")

def is_involutory(a, tol=1e-12):
    return is_square(a) and is_eye(a @ a, tol=tol)

def is_antiinvolutory(a, tol=1e-12):
    return is_square(a) and is_eye(-a @ a, tol=tol)

def is_complex(a):
    a = np.asanyarray(a)
    return np.issubdtype(a.dtype, np.complexfloating)

def is_actually_complex(a, tol=1e-12):
    if not is_complex(a):
        return False
    a = np.asarray(a)
    return np.all(np.abs(a.imag) > tol)

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
    a_dg = a.conj().T
    return _sq_matrix_allclose(a, lambda a: (
        a @ a_dg, a_dg @ a
    ), tol)

def is_projection(a, tol=1e-9):
    a = np.asarray(a)
    return _sq_matrix_allclose(a, lambda a: (
        a @ a, a
    ), a.shape[0]*tol)

def is_projection_orthogonal(a, tol=1e-9):
    return is_projection(a, tol=tol) and is_hermitian(a, tol)

def is_orthogonal_eig(eigs, tol=1e-12):
    """ Check if the given eigenvalues are compatible with the original matrix being orthogonal. Note this is only a *necessary* condition. """
    if not allclose0(abs(eigs) - 1, tol):
        return False
    # find pairs of eigenvalues that are complex conjugates
    found = np.zeros_like(eigs, dtype=bool)
    for i, e in enumerate(eigs):
        if found[i]:
            continue
        conj = np.isclose(e.conj() - eigs[~found], 0)
        if not any(conj):
            return False
        # remove e and its conjugate
        found[np.where(~found)[0][np.argmax(conj)]] = True
        found[i] = True
    return True

def is_diag(a, diag=None, tol=1e-12):
    # return _sq_matrix_allclose(a, lambda a: (
    #     np.diag(np.diag(a)), a
    # ), rtol=rtol, atol=atol)
    if 'scipy' in sys.modules and sp.issparse(a):
        a = sp.dia_array(a)
        return set(a.offsets) == {0}
    a = np.asarray(a)
    n, m = a.shape
    if a.ndim != 2 or n != m:
        raise ValueError(f"Expected square matrix, got {a.shape}")
    if n == 1:
        return diag is None or abs(a[0,0] - diag) <= tol
    if abs(a[0,-1]) > tol:  # shortcut
        return False

    # a[np.isnan(a)] = 0
    if diag is not None and abs(a[-1,-1] - diag) > tol:
        return False
    a = a.reshape(-1)[:-1].reshape(n-1, n+1)  # [:-1] removes a[-1,-1]
    if diag is not None and not allclose0(a[:,0] - diag, tol=tol):
        return False
    a = a[:,1:]  # remove diagonal
    return allclose0(a, tol=tol)

def is_eye(a, tol=0):
    return is_diag(a, 1, tol=tol)

def is_square(a):
    if not hasattr(a, 'shape'):  # don't hide a sparse matrix inside a numpy array
        a = np.asarray(a)
    return a.ndim >= 2 and a.shape[-2] == a.shape[-1]

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

def tf(D, T, T_ = None, is_unitary = True, reverse=False):
    """
    Batched version of T @ D @ T^{-1} where D is diagonal.
    If `reverse=True`, calculate instead T^{-1} @ D @ T.
    """
    if T_ is None:
        if is_unitary:
            T_ = np.swapaxes(T, -2, -1).conj()
        else:
            T_ = np.linalg.inv(T)
    if reverse:
        T, T_ = T_, T
    return T @ (D[..., None] * T_)

def matfunc(A, f):
    D, T = eig(A)
    D = np.asarray(f(D))
    if not is_complex(T) and allclose0(D.imag):
        D = D.real
    return tf(D, T, is_unitary=False)

try:
    from scipy.linalg import expm as matexp
    from scipy.linalg import logm as _matlog
    # for use when importing utils
    from scipy.linalg import fractional_matrix_power as matpow
    from scipy.linalg import sqrtm as matsqrt

    def matlog(A, base=np.e):
        return _matlog(A) / np.log(base)
except ImportError:
    def matexp(A):
        return matfunc(A, np.exp)

    def matlog(A, base=np.e):
        return matfunc(A, lambda x: np.log(x) / np.log(base))

    def matpow(A, n):
        return matfunc(A, lambda x: x**n)

    def matsqrt(A):
        return matfunc(A, np.sqrt)

def matexp_series(A):
    return np.eye(A.shape[0]) + series(lambda n, A_pow: A_pow @ A / n, start_value=A, start_index=1)

def matfunch(A, f):
    D, U = eigh(A)
    D = np.asarray(f(D.astype(complex)))
    return tf(D, U)

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
    return tf(S, V, Vh)

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
            # singular or not square
            pass
    return pinv(A)

def adjugate(a):
    """Return the adjugate of a matrix"""
    # C = np.zeros_like(a)  # cofactor matrix
    # for i,j in shape_it(C):
    #     minor = np.delete(np.delete(a, i, axis=0), j, axis=1)
    #     C[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
    # return C.T
    det_a = np.linalg.det(a)
    if det_a == 0:
        return None
    return inv(a) * det_a

def normalize(a, p=2, axis=0):
    """
    Normalize a vector (or tensor of vectors). For np.ndarray, operates *inplace* and returns the same object.
    """
    if is_complex(a):
        a = np.asarray(a, dtype=complex)
    else:
        a = np.asarray(a).real.astype(float)  # converting complex -> float *and* int -> float
    if a.shape == ():
        return a/np.linalg.norm(a)
    a /= np.linalg.norm(a, ord=p, axis=axis, keepdims=True)
    return a

def kron(A, B, op: np.ufunc=np.multiply):
    A, B = np.asarray(A), np.asarray(B)
    if A.ndim == 1:
        A = A[None,:]
    if B.ndim == 1:
        B = B[None,:]
    assert A.ndim == 2 and B.ndim == 2, f"Invalid dimensions: {A.shape}, {B.shape}"
    res = op.outer(A, B).transpose([0,2,1,3]).reshape(A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])
    if res.shape[0] == 1:
        return res.ravel()
    return res

def kron_eye(d: int, a: np.ndarray, back=False, allow_F=False):
    """ Faster version to calculate np.kron(np.eye(d), a).
    Remark: `np.kron(np.eye(d), a)` (back=False) is much faster than `np.kron(a, np.eye(d))` when enforcing C-layout.
    """
    rows, cols = a.shape
    if back:
        order = 'F' if allow_F else 'C'
        res = np.zeros((rows, d, cols, d), order=order, dtype=a.dtype)
        for i in range(d):
            res[:,i,:,i] = a
    else:
        order = 'C'
        res = np.zeros((d, rows, d, cols), dtype=a.dtype)
        for i in range(d):
            res[i,:,i,:] = a
    return res.reshape(d*rows, d*cols, order=order)

class Sn:
    """
    Symmetric group of order n.
    """
    def __init__(self, n):
        self.n = n

    def sample(self, size=()):
        n = self.n
        return size_samples(lambda: np.random.permutation(n), size)

    def __contains__(self, x):
        return len(x) == self.n and set(x) == set(range(self.n))

    def __iter__(self):
        return itertools.permutations(range(self.n))

    def __len__(self):
        return factorial(self.n)

    def __repr__(self):
        return f'S_{self.n}'

def symmetrization_operator(levels, sign_base=1):
    """
    Returns the symmetrization operator for quantum systems with levels specified by `levels`. Set `sign_base = -1` for antisymmetrization.
    E.g. `assert np.allclose(symmetrization_operator([2,2]), (II + SWAP)/2)`.
    """
    if is_int(levels):
        levels = [2]*int(levels)  # default to qubits
    if sign_base == -1 and all(l == 2 for l in levels) and len(levels) > 2:
        warnings.warn("The antisymmetrization operator for more than 2 qubits is always the zero matrix :)")
    n = len(levels)
    return sum(
        permutation_sign(p, sign_base) * permutation_matrix(p, levels) for p in Sn(n)
    ) / factorial(n)

def permutation_matrix(perm, shape=None):
    """
    Returns a permutation matrix for a given permutation `perm` and shape `shape`. For example,
    the permutation [1,0,2] of three two-level systems can be obtained with `permutation_matrix([1,0,2], [2,2,2])`,
    or, using the short hand for `shape`, with `permutation_matrix([1,0,2], 2)`.

    Parameters:
        perm (list[int]): The permutation of the indices.
        shape (None | int | list[int]): The dimensions of each subsystem, in the original order.
    """
    if shape is None:
        # permute the identity
        return np.eye(len(perm))[perm]
    if is_int(shape):
        shape = int(shape)
        dims  = shape**len(perm)
        shape = [shape]*len(perm)
    elif is_iterable(shape):
        shape = list(shape)
        dims  = prod(shape)
    else:
        raise ValueError(f"Invalid `shape` argument: {shape}")
    assert len(perm) == len(shape), f"Invalid permutation: {perm}"

    idcs = np.unravel_index(np.arange(dims), shape)
    new_idcs = tuple(idcs[perm[j]] for j in range(len(perm)))
    matrix = np.zeros([shape[j] for j in perm] + shape)
    matrix[new_idcs + idcs] = 1
    return matrix.reshape(dims, dims)

def permutation_sign(p, base=-1):
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

def matmoment(A, k, kind='spectral', trace=True, normalized=False, raw=False):
    """
    Matrix moment of order `k` of a batch of iid. matrices.

    Parameters
    ----------
        A (np.ndarray): The batch of matrices.
        k (int): The order of the moment.
        kind (str): The kind of moment to compute. Can be 'spectral', 'absolute', 'power'.
            - `kind = 'spectral'` computes $\\mathbb E[Tr(A^k)]$
            - `kind = 'absolute'` computes $\\mathbb E[|Tr(A^k)|]$
            - `kind = 'power'` computes $\\mathbb E[|Tr(A)|^k]$
        trace (bool): If False, returns above without the trace operation (default: True).
        normalized (bool): If True, normalize by the dimension after the tracing (or where it would be if `trace = False`), e.g. $\\mathbb E[|Tr(A^k)/d|]$ or $\\mathbb E[|Tr(A)/d|^k]$.
        raw (bool): If True, return the individual results of the moment instead of their mean.
    """
    assert kind in ['spectral', 'absolute', 'power'], f"Unknown kind '{kind}'."
    if kind == 'power':
        vals = matmoment(A, 1, trace=trace, kind='absolute', normalized=normalized, raw=True)**k
        return np.mean(vals, axis=0) if not raw else vals
    A = np.asarray(A)
    assert is_square(A), f"Expected square matrix, got {A.shape}"
    assert A.ndim == 3, f"Expected 3D array, got {A.shape}"
    if k == 1:
        vals = [np.trace(x) for x in A] if trace else A
    elif trace and k == 2:
        vals = [trace_product(x.T.conj(), x) for x in A]
    elif k == 0:
        if trace:
            vals = np.zeros(A.shape[0]) + A.shape[1]
        else:
            vals = [np.eye(A.shape[1])] * A.shape[0]
    else:
        tracef = np.trace if trace else lambda x: x
        vals = [tracef(reduce(np.matmul, [x]*k)) for x in A]
    if kind == 'absolute':
        vals = np.abs(vals)
    if not raw:
        vals = np.mean(vals, axis=0)
    if normalized:
        vals = np.asarray(vals) / A.shape[1]
    return vals

def commutator(A, B):
    return A @ B - B @ A

def commutes(A, B, tol=1e-10):
    """ Check if two matrices commute. """
    return allclose0(commutator(A, B), tol=tol)

def anticommutator(A, B):
    return A @ B + B @ A

def anticommute(A, B, tol=1e-10):
    """ Check if two matrices anticommute. """
    return allclose0(anticommutator(A, B), tol=tol)

def trace_product(A, B):
    """Hilbert-Schmidt product or trace inner product of two matrices."""
    if A.ndim == 2:
        return A.ravel().conj() @ B.ravel()  # Liouville space inner product <<A|B>>
    return np.einsum('...ij,...ij->...', U.conj(), V)
    return np.trace(A.T.conj() @ B)

Hilbert_schmidt_inner_product = trace_product

def svs(A, is_hermitian=False):
    """ Compute the singular values of a matrix. """
    if is_hermitian:
        return np.abs(eigvalsh(A))
    return svd(A, compute_uv=False)

def spectral_norm(A, is_hermitian=False):
    """Spectral norm or Schatten ∞-norm of a matrix."""
    return max(svs(A, is_hermitian))
    return np.linalg.norm(A, ord=2)

def spectral_radius(A, is_hermitian=False):
    """Spectral radius of a matrix."""
    if is_hermitian:
        return max(svs(A, True))
    return max(np.abs(eigvals(A)))

def Schatten_norm(A, p, is_hermitian=False):
    """Schatten norm of a matrix."""
    return sum(svs(A, is_hermitian)**p)**(1/p)

def Lpq_norm(A, p, q):
    """L_{p,q} norm of a matrix."""
    return sum(sum(np.abs(A)**p)**(q/p))**(1/q)

def trace_norm(A, is_hermitian=False):
    """Trace norm, nuclear norm, or Schatten 1-norm of a matrix."""
    return sum(svs(A, is_hermitian))  # Schatten 1-norm
    return np.linalg.norm(A, ord='nuc')
    return np.trace(matsqrt(A.T.conj() @ A))

nuclear_norm = trace_norm

def frobenius_norm(A):
    """Frobenius norm, Hilbert-Schmidt, L_{2,2} or Schatten 2-norm of a matrix."""
    return np.linalg.norm(A)                # defaults to Frobenius norm (ord='fro')
    return sqrt(trace_product(A, A).real)   # Hilbert-Schmidt norm
    return sqrt(np.sum(np.abs(A)**2))       # L_{2,2} norm
    return Schatten_norm(A, 2)              # Schatten 2-norm

hilbert_schmidt_norm = frobenius_norm

def spectral_gap(T, is_normal=False):
    """ Difference between highest and second-highest eigenvalue. """
    if is_normal:
        D = np.linalg.svd(T, compute_uv=False)
    else:
        D = np.abs(eigvals(T))
        D = sorted(D, reverse=True)
    return abs(D[0]) - abs(D[1])

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
            D_sqrt = np.sqrt(D)
            assert np.all(D > 0), f"Matrix is not invertible: {D}"
            J     = tf(D_sqrt, U_S)
            J_inv = tf(1/D_sqrt, U_S)
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
        J = tf(S, Vh, reverse=True)
        return U, J
    elif kind == 'left':
        K = tf(S, W)
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

    def so(n, sparse=False, normalize=False, output_hermitian=False, as_eigen=False):
        """
        The Lie algebra associated with the Lie group SO(n). Returns the n(n-1)/2 generators (antisymmetric matrices) of the group.

        Parameters
        ----------
            n (int): The dimension of the matrices.
            sparse (bool, optional): If True, return a sparse representation of the matrices (default: False).
            normalize (bool, optional): If True, normalize the matrices to have norm 1 (default: False).
            output_hermitian (bool, optional): If True, output the Hermitian basis elements `ig` instead.
            as_eigen (bool, optional): If True, return each generators already diagonalized as tuple `(D, U)`

        Returns
        -------
        list[ np.ndarray | scipy.sparse.csr_array ]
            A list of `n(n-1)/2` matrices that form a basis of the Lie algebra.
        """
        basis = []
        if as_eigen:
            if sparse:
                Ubase = sp.eye(n, dtype=complex)
                Dbase = sp.lil_array(n, dtype=complex)
            else:
                Ubase = np.eye(n, dtype=complex)
                Dbase = np.zeros(n, dtype=complex)

            fs2 = 1/sqrt(2)
            ifs2 = 1j*fs2
            if output_hermitian:
                fs2, ifs2 = ifs2, fs2
            ev = ifs2 if normalize else 1j
            for i in range(n):
                for j in range(i+1, n):
                    D = Dbase.copy()
                    D[i] = ev
                    D[j] = -ev

                    U = Ubase.copy()
                    U[i,i] = U[i,j] = fs2
                    U[j,i] = ifs2
                    U[j,j] = -ifs2

                    if sparse:
                        basis.append((sp.csr_array(D), sp.csr_array(U)))
                    else:
                        basis.append((D, U))
        else:
            dtype = complex if output_hermitian else float
            if sparse:
                base = sp.lil_array((n,n), dtype=dtype)
            else:
                base = np.zeros((n,n), dtype=dtype)
            ev = 1.0
            if normalize:
                ev /= sqrt(2)
            if output_hermitian:
                ev *= 1j
            for i in range(n):
                for j in range(i+1, n):
                    m = base.copy()
                    m[i,j] = ev
                    m[j,i] = -ev
                    basis.append(m)
        if sparse:
            basis = [sp.csr_array(m) for m in basis]
        return basis

    def su(n, include_identity=False, output_hermitian=True, sparse=False, normalize=False):
        """ The Lie algebra associated with the Lie group SU(n). Returns the n^2-1 generators (traceless Hermitian matrices) of the group. Use `output_hermitian = True` and `include_identity = True` to return a complete orthogonal basis of hermitian `n x n` matrices.

        Parameters:
            n (int): The dimension of the matrices.
            include_identity (bool, optional): If True, include the identity matrix in the basis (default: False).
            output_hermitian (bool, optional): If True, output the Hermitian basis elements `ig` instead.
            sparse (bool, optional): If True, return a sparse representation of the matrices (default: False).
            normalize (bool, optional): If True, normalize the matrices to have norm 1 (default: False).

        Returns
        -------
        list[ np.ndarray | scipy.sparse.csr_array ]:
            A list of `n^2-1` matrices that form a basis of the Lie algebra.
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
            identity = sp.eye(n) if sparse else np.eye(n)
            if normalize:
                # factor 2 to get norm sqrt(2), too
                identity = sqrt(2/n) * identity
            if not output_hermitian:
                identity = 1j*identity
            basis.append(identity)

        # su have norm sqrt(2) by default
        el = 1/sqrt(2) if normalize else 1
        eli = 1j*el
        if not output_hermitian:
            el, eli = eli, el

        # Generate the off-diagonal matrices
        for i in range(n):
            for j in range(i+1, n):
                m = base.copy()
                m[i,j] = el
                m[j,i] = el
                basis.append(m)

                m = base.copy()
                m[i, j] = -eli
                m[j, i] = eli
                basis.append(m)

        # Generate the diagonal matrices
        for i in range(1,n):
            m = base.copy()
            for j in range(i):
                m[j,j] = el
            m[i,i] = -i*el
            if i > 1:
                m = sqrt(2/(i*(i+1))) * m
            basis.append(m)

        if sparse:
            basis = [sp.csr_array(m) for m in basis]
        return basis

    class LieGroupElement:
        """ A class representing an element of a Lie group."""
        def __init__(self, G, convention, is_generator=True):
            """ The generator `G` is expected as hermitian. """
            self.c = convention
            self.is_generator = is_generator

            if isinstance(G, tuple):
                self._D, self._U = G
                self._G = None
                assert self.D.shape[0] == self.U.shape[0] == self.U.shape[1], f"Invalid shapes: {self.D.shape}, {self.U.shape}"
                if not is_generator:
                    self._logD()
                    self.is_generator = True
            else:
                self._G = G
                self._D, self._U = None, None

        def _diagonalize(self):
            if self._D is None or self._U is None:
                if self.is_generator:
                    self._D, self._U = eigh(self._G)
                else:
                    T, Q = schur(self._G)  # self._G is a unitary
                    assert is_diag(T), f"Non-diagonalizable matrix: {T}"
                    self._D, self._U = np.diag(T), Q
                    self._logD()
                    self.is_generator = True
                self._G = None

        def _logD(self):
            self._D = -1j/self.c*np.log(self._D)
            assert allclose0(self.D.imag), f"Non-real eigenvalues in generator: {self._D}"
            self._D = self._D.real

        @property
        def D(self):
            if self._D is None:
                self._diagonalize()
            return self._D

        @property
        def U(self):
            if self._U is None:
                self._diagonalize()
            return self._U

        @property
        def G(self):
            if self._G is None or not self.is_generator:
                self._G = tf(self.D, self.U)
            return self._G

        @property
        def n(self):
            if self._D is not None:
                return len(self._D)
            return len(self._G)

        def __call__(self, phi):
            return tf(np.exp(1j*self.c*phi*self.D), self.U)

        def __matmul__(self, other):
            return LieGroupElement(self(1) @ other(1), self.c, is_generator=False)

        def __pow__(self, a):
            return LieGroupElement((a*self.D, self.U), self.c)

        def __repr__(self):
            pre = f'exp({1j}*'
            if self.c != 1:
                pre += f'({self.c})*'
            pre += 'phi*'
            G_str = repr(self.G)[6:]
            return pre + G_str.replace('\n', '\n' + ' '*(len(pre)-6))

    class LieGroup(Group):
        def __init__(self, generators, convention=1):
            els = [LieGroupElement(G, convention) for G in generators]
            n = els[0].n
            super().__init__(els, identity=LieGroupElement((np.zeros(n, dtype=complex), np.eye(n)), convention))

        def op(self, x: LieGroupElement, y: LieGroupElement):
            return x @ y

        def pow(self, x: LieGroupElement, a):
            return x**a

        def __contains__(self, x: LieGroupElement):
            return NotImplemented

    class SU(LieGroup):
        """ Special unitary group. Returns n^2-1 callables that take an angle and return the corresponding unitary matrix """
        def __init__(self, n):
            self.n = n
            super().__init__(su(n, output_hermitian=True), convention=-1/2)

        def sample(self):
            U = random_unitary(self.n)
            return det(U)**(-1/self.n) * U

        def __repr__(self):
            return f'SU({self.n}) ({len(self)} dimensions)'

        def __contains__(self, x):
            if isinstance(x, LieGroupElement):
                return x.D.shape == (self.n, self.n)
            elif isinstance(x, np.ndarray):
                return x.shape == (self.n, self.n) and is_unitary(x) and abs(det(x) - 1) < 1e-12
            raise ValueError(f"Invalid type: {type(x)}")

    class SO(LieGroup):
        """ Special orthogonal group. Returns n(n-1)/2 callables that take an angle and return the corresponding orthogonal matrix """
        def __init__(self, n):
            self.n = n
            super().__init__(so(n, output_hermitian=True, as_eigen=True))

        def sample(self):
            O = random_orthogonal(self.n)
            det_O = det(O)
            if det_O < 0:
                O[:,0] *= -1
            return O

        def __repr__(self):
            return f'SO({self.n}) ({len(self)} dimensions)'

        def __contains__(self, x):
            if isinstance(x, LieGroupElement):
                return x.D.shape == (self.n, self.n) and is_orthogonal_eig(x.D)
            elif isinstance(x, np.ndarray):
                return x.shape == (self.n, self.n) and is_orthogonal(x) and abs(det(x) - 1) < 1e-12
            raise ValueError(f"Invalid type: {type(x)}")

    def SO_old(n, sparse=False):
        """ Special orthogonal group. Returns n(n-1)/2 functions that take an angle and return the corresponding real rotation matrix """
        eye = sp.eye if sparse else np.eye
        def rotmat(i, j, phi):
            a = eye(n)
            cp, sp = cos(phi), sin(phi)
            a[i,i] = cp
            a[j,j] = cp
            a[i,j] = -sp
            a[j,i] = sp
            return a
        return [lambda phi: rotmat(i, j, phi) for i,j in itertools.combinations(range(n), 2)]

def generate_recursive(stubs, n, basis, extend_fn):
    if n <= 1:
        return stubs
    return (m for s in stubs for m in generate_recursive([extend_fn(s, b) for b in basis], n-1, basis, extend_fn))

def pauli_basis(n, kind='np', normalize=False):
    """ Generate the pauli basis of hermitian 2**n x 2**n matrices. This basis is orthonormal and, except for the identity, traceless. They are also unitary and therefore involutory.

    E.g. for n = 2, the basis is [II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ]

    Parameters:
        n (int): Number of qubits
        kind (str): 'np' for numpy arrays (default), 'sp' for scipy sparse matrices, or 'str' for strings
        normalize (bool): Whether to normalize the basis elements (default False)

    Returns
    -------
    list[ np.ndarray | scipy.sparse.csr_array | str ]
        The pauli basis
    """
    if kind == 'str':
        norm_str = f"{1/sqrt(2**n)}*" if normalize else ""
        return [norm_str + ''.join(i) for i in itertools.product(['I', 'X', 'Y', 'Z'], repeat=n)]
        # basis = ['I', 'X', 'Y', 'Z']
        # res = generate_recursive(basis, n, basis, lambda a, b: a+b, normalize)
        # if normalize:
        #     res = [f'{norm_str}({i})' for i in res]
        # return res
    elif kind not in ['np', 'sp']:
        raise ValueError(f"Unknown kind: {kind}")

    if n > 8:
        warnings.warn(f"Generating {2**(2*n)} {2**n}x{2**n} Pauli basis matrices (n = {n}) may take a long time.", stacklevel=2)
    basis = su(2, include_identity=True, sparse=kind == 'sp')
    stubs = [m/sqrt(2**n) for m in basis] if normalize else basis
    extend_fn = sp.kron if kind == 'sp' else np.kron
    return generate_recursive(stubs, n, basis, extend_fn=extend_fn)

# from https://docs.pennylane.ai/en/stable/code/api/pennylane.pauli_decompose.html
def pauli_decompose(H, as_str=False, eps=1e-5):
    r"""Decomposes a Hermitian matrix into a linear combination of Pauli operators.

    Parameters:
        H (ndarray): Hermitian matrix of shape `(2**n, 2**n)`
        eps (float): Threshold to include a term in the decomposition. Set to 0 to include all terms.

    Returns
    -------
    tuple[ list[float], list[str] ] | str
        The coefficients and the Pauli operator strings

    Example
    >>> H = np.array([[-2, -2+1j, -2, -2], [-2-1j,  0,  0, -1], [-2,  0, -2, -1], [-2, -1, -1,  0]])
    >>> pauli_decompose(H)
    ([-1.0, -1.5, -0.5, -1.0, -1.5, -1.0, -0.5, 1.0, -0.5, -0.5],
     ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XZ', 'YY', 'ZX', 'ZY'])
    """
    H = np.asarray(H)
    n = int(log2(H.shape[0]))
    N = 2**n

    if H.shape != (N, N):
        raise ValueError(f"The matrix should have shape (2**n, 2**n), for a number of qubits n>=1, but is {H.shape}")

    assert is_hermitian(H), f"The matrix is not Hermitian:\n{H}"

    obs_lst = []
    coeffs = []

    for term, basis_matrix in zip(pauli_basis(n, kind='str'), pauli_basis(n, kind='np')):
        coeff = trace_product(basis_matrix, H).real / N  # project H onto the basis matrix

        if abs(coeff) >= eps:
            coeffs.append(coeff)
            obs_lst.append(term)

    if as_str:
        return str_from_pauli(coeffs, obs_lst, precision=5 if eps == 0 else ceil(-log10(eps)))
    return coeffs, obs_lst

def str_from_pauli(coeffs, obs_lst, precision=5):
    """ Convert a Pauli decomposition to a string. """
    return ' + '.join([f"{coeff:.{precision}f}*{obs}" for coeff, obs in zip(coeffs, obs_lst)])

##############
### Random ###
##############

def random_vec(size, params=(0,1), complex=False, kind='normal'):
    if kind == 'normal':
        rng = np.random.normal
    elif kind == 'uniform':
        rng = np.random.uniform
    else:
        raise ValueError(f"Unknown kind '{kind}'.")
    if complex:
        return rng(*params, size=size) + 1j*rng(*params, size=size)
    return rng(*params, size=size)

def random_square(size, params=(0,1), complex=False, kind='normal'):
    if not hasattr(size, '__len__'):
        size = (size,)
    size += (size[-1],)  # repeat last dimension
    return random_vec(size, params=params, complex=complex, kind=kind)

def random_symmetric(n, params=(0,1)):
    """
    Sample a random symmetric matrix from the Gaussian Orthogonal Ensemble (GOE).
    """
    a = np.diag(random_vec(n, params=params, complex=False))
    a[np.triu_indices(n, 1)] = random_vec(n*(n-1)//2, params=params, complex=False)
    a[a == 0] = a.T[a == 0]
    return a

def random_orthogonal(n, size=(), params=(0,1)):
    """
    Sample a random orthogonal matrix according to the real Haar measure.
    """
    a = random_vec(size + (n,n), params=params, complex=False)
    return np.linalg.qr(a)[0]

def random_hermitian(n, size=(), std=1, normalized=True):
    """
    Sample a random Hermitian matrix from the Gaussian Unitary Ensemble (GUE).
    """
    if not hasattr(size, '__len__'):
        size = (size,)
    if normalized:
        std /= sqrt(n)
    if n > 15:
        a = np.zeros(size + (n,n), dtype=complex)
        diags = random_vec(size + (n,), params=(0, std), complex=False).astype(complex)
        n_tri  = n*(n-1)//2  # number of triangular elements
        trius = random_vec(size + (n_tri,), params=(0, std/sqrt(2)), complex=True)
        is_triu = ~np.tri(n, dtype=bool)
        is_tril = np.tri(n, k=-1, dtype=bool)
        for i in shape_it(size):
            H = np.diag(diags[i])
            H[is_triu] = trius[i]
            H[is_tril] = H.T.conj()[is_tril]
            a[i] = H
        return a
    # equivalent, but faster for small matrices
    a = random_vec(size + (n,n), params=(0, std), complex=True, kind='normal')
    return (a + np.moveaxis(a, -1, -2).conj())/2

def random_unitary(n, size=(), kind='haar'):
    """
    Sample a random unitary.
    - `kind = 'haar'` samples from the complex Haar measure (default).
    - `kind = 'gue'` samples a GUE matrix and returns its eigenbasis. Also Haar-distributed, but slower than above.
    - `kind = 'polar'` is the fastest for very small matrices.
    """
    if not hasattr(size, '__len__'):
        size = (size,)
    if kind == 'haar':
        return random_isometry(n, n, size=size)
    elif kind == 'gue':
        return eigh(random_hermitian(n, size=size))[1]
    elif kind == 'polar':  # fastest for very small and slowest for very large matrices
        A = random_square(n, complex=True, kind='normal')
        D, U = eigh(A.T.conj() @ A)
        D_sqrt = np.sqrt(D)
        J_inv = tf(1/D_sqrt, U)
        return A @ J_inv
    else:
        raise ValueError(f"Unknown kind '{kind}'.")

def random_isometry(n, m, size=()):
    assert n >= m, f"n must be >= m, but got {n} < {m}"
    A = random_vec(size + (n,m), complex=True, kind='normal')
    Q, R = np.linalg.qr(A)
    R_ = np.zeros(size + (m,), dtype=complex)
    for i in shape_it(size):
        R_[i] = np.diag(R[i])
    L = R_ / np.abs(R_)
    return Q * L[..., None, :]

def unitary_noise(d, s, size=()):
    H = random_hermitian(d, size=size)
    return matexp(1j*s*H)

def random_psd(n, params=(0,1), complex=True):
    params = (params[0], sqrt(params[1]))  # eigs scale with variance
    a = random_square(n, params=params, complex=complex, kind='normal')
    return a @ a.conj().T

def random_normal(n, params=(0,1), complex=True):
    U = random_unitary(n)
    D = random_vec(U.shape[0], params=params, complex=complex, kind='normal')
    return tf(D, U)

def random_projection(n, rank=None, orthogonal=True, complex=True, kind='fast'):
    """
    Sample a random projection matrix P^2 = P. If `orthogonal = True`, the sample is also hermitian.
    - `kind = uniform` samples uniformly from the space of projectors (only implemented for orthogonal projections).
    - `kind = 'fast'` is faster, especially for rank << size (default).
    """
    if rank is None:
        rank = np.random.randint(1, n) #+orthogonal)  # rank == n is always orthogonal (identity)
    else:
        rank = min(rank, n)

    if kind == 'fast':
        # much faster for rank << size
        A = random_vec((n, rank), complex=complex)
        if orthogonal:
            B = A.conj().T
        else:
            B = random_vec((rank, n), complex=complex)
        return A @ sinv(B @ A, tol=n*1e-9) @ B
    elif kind == 'uniform':
        if orthogonal:
            # P^2 = P and P = P^H
            if complex:
                U = random_unitary(n)
            else:
                U = random_orthogonal(n)
            D = np.random.permutation([1]*rank + [0]*(n-rank))
            return tf(D, U)
        else:
            # only P^2 = P
            A = random_square(n, complex=complex)
            D = np.random.permutation([1]*rank + [0]*(n-rank))
            return tf(D, A, is_unitary=False)
    else:
        raise ValueError(f"Unknown kind '{kind}'.")

def random_involutory(size, kind='uniform'):
    P = random_projection(size, orthogonal=True, complex=True, kind=kind)
    return 2*P - np.eye(size)

########################
### Integer matrices ###
########################

def inv_q(A, q):
    """Matrix inverse in F_q"""
    # determinant
    if len(A) == 2:
        det = (A[0][0]*A[1][1] - A[0][1]*A[1][0])
    else:
        det = np.round(np.linalg.det(A))
    det_mod = int(det) % q
    if det_mod == 0:
        return None
    # determinant inverse
    det_inv = pow(det_mod, -1, q)
    if det_inv is None:
        return None
    # adjugate matrix
    if len(A) == 2:
        adjugate = np.array([[A[1][1], -A[0][1]], [-A[1][0], A[0][0]]]) % q
    else:
        adjugate = np.round(np.linalg.inv(A) * det).astype(int) % q
    # inverse matrix
    A_inv = adjugate*det_inv % q
    return A_inv

class SL(Group):
    """ Special linear group of nxn matrices over F_q. """
    def __init__(self, n, q):
        self.n = n
        self.q = q
        super().__init__(self.generate(), identity=self.cannonical(np.eye(n, dtype=int)))

    def generate(self):
        sl = set()
        for matrix in itertools.product(range(self.q), repeat=self.n*self.n):
            matrix = np.array(matrix).reshape(self.n, self.n)
            if np.round(np.linalg.det(matrix)) % self.q == 1:
                sl.add(self.cannonical(matrix))
        return sl

    def op(self, A, B):
        A, B = np.asarray(A), np.asarray(B)
        res = (A @ B) % self.q
        return self.cannonical(res)

    def inv(self, A):
        return self.cannonical(inv_q(A, self.q))

    def cannonical(self, A):
        return tuple(map(tuple, A))

    def __repr__(self):
        return f"SL({self.n}, {self.q}) ({len(self)} elements)"

class PSL(SL):
    """ Projective special linear group of nxn matrices over F_q. """
    def __init__(self, n, q):
        self.lambdas = mod_roots(1, n, q)
        super().__init__(n, q)

    def cannonical(self, A):
        As = []
        for l in self.lambdas:
            f = tuple if l == 1 else lambda x: tuple(map(lambda y: (l*y) % self.q, x))
            As.append(tuple(map(f, A)))
        return min(As)

    def __repr__(self):
        return "P" + super().__repr__()

def conjugacy_classes(G: Group):
    conjugacy_classes = []
    for h in G:
        if not any(h in c for c in conjugacy_classes):
            conjugacy_class = set()
            for g in G:
                g_inv = G.inv(g)
                if g_inv is not None:
                    conjugate = G.op(g_inv, G.op(h, g))
                    if not any(conjugate in c for c in conjugacy_classes):
                        conjugacy_class.add(conjugate)
            conjugacy_classes.append(conjugacy_class)

    class_info = []
    for cls in conjugacy_classes:
        re = next(iter(cls))
        class_info.append({
            "size": len(cls),
            "representative": re
        })
    info = {
        "group_order": len(G),
        "num_classes": len(conjugacy_classes),
        "classes": class_info
    }
    return conjugacy_classes, info