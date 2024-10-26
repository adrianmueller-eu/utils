import sys, warnings
import numpy as np
from math import factorial, sqrt, ceil
from itertools import combinations, chain, permutations
import scipy.sparse as sp
from numpy.random import randint
from .models import Polynomial

sage_loaded = False
try:
    from sage.all import *
    sage_loaded = True
except ImportError:
    pass

##############
### Basics ###
##############

# e.g. series(lambda n, _: 1/factorial(2*n)) + series(lambda n, _: 1/factorial(2*n + 1))
def series(f, start_value=0, start_index=0, eps=sys.float_info.epsilon, max_iter=100000, verbose=False, tqdm=False):
    """ Calculate the series $start_value + \\sum_{n=start_index+1}^{\\infty} f(n, f(n-1, ...))$. Throws an error if the series doesn't converge.

    Parameters
        f (function): A function that takes two arguments, the current iteration `i` and the last term `term`, and returns the next term in the series.
        start_value (float | np.ndarray, optional): The value of the series at `start_index` (default: 0).
        start_index (int, optional): The index at which to start the series (default: 0).
        eps (float, optional): The precision to which the series should be calculated (default: `sys.float_info.epsilon`).
        max_iter (int, optional): The maximum number of iterations (default: 100000).
        verbose (bool, optional): If True, print the current iteration and the current value of the series (default: False).
        tqdm (tqdm.tqdm, optional): Use tqdm for progress bar (default: False). You might give a custom tqdm object.

    Returns
        float | np.ndarray: The value of the series.

    Examples:
        >>> series(lambda n, _: 1/factorial(2*n), 1) + series(lambda n, _: 1/factorial(2*n + 1), 1)
        2.7182818284590455
    """
    if not tqdm:
        def tq(x):  # dummy function
            return x
    elif not callable(tqdm):
        from tqdm.auto import tqdm as tq
    else:
        tq = tqdm

    if not np.isscalar(start_value):
        start_value = np.array(start_value)
    res = start_value
    term = res
    for i in tq(range(start_index+1, max_iter)):
        term = f(i, term)
        res += term
        change = np.nansum(np.abs(term))
        if verbose:
            print(f"Iteration {i}:", res, term)
        if change < eps:
            return res  # return when converged
        if np.max(res) == np.inf:
            break

    raise ValueError(f"Series didn't converge after {max_iter} iterations! Error: {np.sum(np.abs(term))}")

def sequence(f, start_value=0, start_index=0, eps=sys.float_info.epsilon, max_iter=100000, verbose=False):
    """ Calculate the sequence $[f(start_index), f(start_index+1), ...]$ until it converges or the maximum number of iterations is reached. Then return the last term of the sequence.

    Parameters
        f (function): A function that takes two arguments, the current iteration `i` and last term `last_term`, and returns the next term in the sequence.
        start_value (float | np.ndarray, optional): The value of the series at `start_index` (default: 0).
        start_index (int, optional): The index at which to start the series (default: 0).
        eps (float, optional): The precision to which the series should be calculated (default: `sys.float_info.epsilon`).
        max_iter (int, optional): The maximum number of iterations (default: 100000).
        verbose (bool, optional): If True, print the current iteration and the current value of the series (default: False).

    Returns
        float | np.ndarray: The value of the series.
    """
    if not np.isscalar(start_value):
        start_value = np.array(start_value)
    last_term = start_value
    for i in range(start_index+1, max_iter):
        current_term = f(i, last_term)
        if verbose:
            print(f"Iteration {i}:", current_term)
        # if it contains inf or nan, we assume divergence
        if np.isinf(current_term).all() or np.isnan(current_term).all():
            print(f"Warning: Sequence diverged after {i} iterations!")
            return current_term
        # if the difference between the last two terms is smaller than eps, we assume convergence
        error = np.sum(np.abs(current_term - last_term))
        if error < eps:
            if verbose:
                print(f"Converged after {i} iterations! Error: {error}")
            return current_term
        last_term = current_term

    print(f"Warning: Sequence didn't converge after {max_iter} iterations! Error: {error}")
    return current_term

def normalize(a, p=2, axis=0, remove_global_phase_if_1D=False):
    if is_complex(a):
        a = np.array(a, dtype=complex)
    else:
        a = np.array(a, dtype=float)
    if a.shape == ():
        return a/np.linalg.norm(a)
    a /= np.linalg.norm(a, ord=p, axis=axis, keepdims=True)
    if len(a.shape) == 1 and remove_global_phase_if_1D and is_complex(a):
        # this works only for a 1D array
        a *= np.exp(-1j*np.angle(a[0]))
    return a

def arctan2(y, x):  # same as np.arctan2
    if x == 0:
        return np.sign(y) * np.pi/2
    if y == 0:
        return (1 - np.sign(x)) * np.pi/2
    if x < 0:
        return np.arctan(y/x) + np.sign(y)* np.pi
    return np.arctan(y/x)
    # if x == 0:
    #     return np.sign(y) * np.pi/2
    # if x < 0:
    #     if y < 0:
    #         return np.arctan(y/x) - np.pi
    #     else:
    #         return np.arctan(y/x) + np.pi
    # return np.arctan(y/x)

def is_odd(x):
    return x % 2 == 1

def is_even(x):
    return x % 2 == 0

#################
### Deg / rad ###
#################

def deg(rad):
    return rad/np.pi*180

def rad(deg):
    return deg/180*np.pi

################
### Matrices ###
################

### property checks

def _sq_matrix_allclose(a, f, rtol=1e-05, atol=1e-08):
    a = np.array(a, copy=False)
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        return False
    a[np.isnan(a)] = 0
    a, b = f(a)
    return np.allclose(a, b, rtol=rtol, atol=atol)

def is_symmetric(a, rtol=1e-05, atol=1e-08):
    return _sq_matrix_allclose(a, lambda a: (
        a, a.T
    ), rtol=rtol, atol=atol)

def is_hermitian(a, rtol=1e-05, atol=1e-08):
    return _sq_matrix_allclose(a, lambda a: (
        a, a.conj().T
    ), rtol=rtol, atol=atol)

def is_antihermitian(a, rtol=1e-05, atol=1e-08):
    return _sq_matrix_allclose(a, lambda a: (
        a, -a.conj().T
    ), rtol=rtol, atol=atol)

def is_orthogonal(a, rtol=1e-05, atol=1e-08):
    return _sq_matrix_allclose(a, lambda a: (
        a @ a.T, np.eye(a.shape[0])
    ), rtol=rtol, atol=atol)

def is_unitary(a, rtol=1e-05, atol=1e-08):
    return _sq_matrix_allclose(a, lambda a: (
        a @ a.conj().T, np.eye(a.shape[0])
    ), rtol=rtol, atol=atol)

def is_involutory(a, rtol=1e-05, atol=1e-08):
    return _sq_matrix_allclose(a, lambda a: (
        a @ a, np.eye(a.shape[0])
    ), rtol=rtol, atol=atol)

def is_complex(a):
    if hasattr(a, 'dtype'):
        if a.dtype == complex:
            return True
        return np.issubdtype(a.dtype, complex)
    return np.iscomplex(a).any()

def is_psd(a, rtol=1e-05, atol=1e-08):
    if not is_hermitian(a, rtol=rtol, atol=atol):
        return False
    eigv = np.linalg.eigvalsh(a)
    return np.all(np.abs(eigv.imag) < atol) and np.all(eigv.real >= -atol)

def is_normal(a, rtol=1e-05, atol=1e-08):
    return _sq_matrix_allclose(a, lambda a: (
        a @ a.conj().T, a.conj().T @ a
    ), rtol=rtol, atol=atol)

def is_projection(a, rtol=1e-05, atol=1e-08):
    return _sq_matrix_allclose(a, lambda a: (
        a @ a, a
    ), rtol=rtol, atol=atol)

def is_projection_orthogonal(a, rtol=1e-05, atol=1e-08):
    return is_projection(a, rtol=rtol, atol=atol) and is_hermitian(a, rtol=rtol, atol=atol)

### matrix functions

try:
    from scipy.linalg import expm as matexp
    from scipy.linalg import logm as _matlog
    from scipy.linalg import sqrtm as matsqrt
    from scipy.linalg import fractional_matrix_power as matpow

    def matlog(A, base=np.e):
        return _matlog(A) / np.log(base)
except:
    def matexp(A):
        # there is a faster method for hermitian matrices
        if is_hermitian(A):
            eigval, eigvec = np.linalg.eigh(A)
            return eigvec @ np.diag(np.power(np.e, eigval)) @ eigvec.conj().T
        # use series expansion
        return np.eye(A.shape[0]) + series(lambda n, A_pow: A_pow @ A / n, start_value=A, start_index=1)

    def matlog(A, base=np.e):
        evals, evecs = np.linalg.eig(A)
        return evecs @ np.diag(np.log(evals.astype(complex)) / np.log(base)) @ evecs.conj().T

    def matpow(A, n):
        evals, evecs = np.linalg.eig(A)
        return evecs @ np.diag(evals.astype(complex)**n) @ evecs.conj().T

    def matsqrt(A, n=2):
        return matpow(A, 1/n)

def permutation_sign(p, base):
    if base == 1:
        return 1
    inversions = sum(i > j for i,j in combinations(p, 2))
    return base**inversions

def immanant(A, char):
    """ Thanks for the cookies! """
    r = range(A.shape[0])
    return sum(permutation_sign(P, char) * np.prod(A[r, P]) for P in permutations(r))

def permanent(A):
    return immanant(A, 1)

def determinant(A):
    # np.prod(np.linalg.eigvals(A))  # O(n^3)
    return immanant(A, -1)           # O(n!)

def commutator(A, B):
    return A @ B - B @ A

def commute(A, B):
    """ Check if two matrices commute. """
    return np.allclose(commutator(A, B), 0)

def anticommutator(A, B):
    return A @ B + B @ A

def anticommute(A, B):
    """ Check if two matrices anticommute. """
    return np.allclose(anticommutator(A, B), 0)

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

# def polar(A, kind='left'):
#     """Polar decomposition of a matrix into a unitary and a PSD matrix: $A = UJ$ or $A = KU$."""
#     if kind == 'left':
#         J = matsqrt(A.T.conj() @ A)
#         U = A @ np.linalg.pinv(J)
#         return U, J
#     elif kind == 'right':
#         K = matsqrt(A @ A.T.conj())
#         U = np.linalg.pinv(K) @ A
#         return K, U
#     raise ValueError(f"Unknown kind '{kind}'.")

# def svd(A):  # np.linalg.svd is faster
#     S, J = polar(A, kind='left')
#     D, T = np.linalg.eig(J)
#     return S @ T, np.diag(D), T.conj().T

### rotation groups

if not sage_loaded:

    def SO(n):
        """ Special orthogonal group. Returns n(n-1)/2 functions that take an angle and return the corresponding real rotation matrix """
        def rotmat(i, j, phi):
            a = np.eye(n)
            a[i,i] = np.cos(phi)
            a[j,j] = np.cos(phi)
            a[i,j] = -np.sin(phi)
            a[j,i] = np.sin(phi)
            return a
        return [lambda phi: rotmat(i, j, phi) for i,j in combinations(range(n), 2)]

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
                print(f"Warning: For `n = {n} > 100`, it is recommended to use `sparse=True` to save memory.")
            base = np.zeros((n,n), dtype=complex)

        basis = []

        # Identity matrix, optional
        if include_identity:
            identity = base.copy()
            for i in range(n):
                identity[i,i] = 1
            if normalize:
                # factor 2 to get norm sqrt(2), too
                identity = np.sqrt(2/n) * identity
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
                m = np.sqrt(2/(i*(i+1))) * m
            basis.append(m)

        if normalize:
            # su have norm sqrt(2) by default
            basis = [m/np.sqrt(2) for m in basis]
        if sparse:
            # convert to csr format for faster arithmetic operations
            return [sp.csr_matrix(m) for m in basis]
        return basis

    def SU(n):
        """ Special unitary group. Returns n^2-1 functions that take an angle and return the corresponding complex rotation matrix """
        generators = su(n)
        def rotmat(G):
            D, U = np.linalg.eigh(G)
            return lambda phi: U @ np.diag(np.exp(-1j*phi/2*D)) @ U.conj().T
        return [rotmat(G) for G in generators]

### random

def random_vec(size, limits=(0,1), complex=False):
    if complex:
        return random_vec(size, limits=limits) + 1j*random_vec(size, limits=limits)
    return np.random.uniform(limits[0], limits[1], size=size)

def random_square(size, limits=(0,1), complex=False):
    if not hasattr(size, '__len__'):
        size = (size, size)
    if size[0] != size[1] or len(size) != 2:
        raise ValueError(f"The shape must be square, but was {size}.")
    return random_vec(size, limits=limits, complex=complex)

def random_symmetric(size, limits=(0,1)):
    a = random_square(size, limits=limits)
    return (a + a.T)/2

def random_orthogonal(size):
    a = random_square(size)
    q, r = np.linalg.qr(a)
    return q

def random_hermitian(size, limits=(0,1)):
    a = random_square(size, limits=limits, complex=True)
    return (a + a.conj().T)/2

def random_unitary(size):
    H = random_hermitian(size)
    return matexp(1j*H)

def random_psd(size, limits=(0,1), complex=True):
    limits = np.sqrt(limits)  # because we square it later
    a = random_square(size, limits=limits, complex=complex)
    return a @ a.conj().T

def random_normal(size, limits=(0,1), complex=True):
    U = random_unitary(size)
    D = np.diag(random_vec(U.shape[0], limits=limits, complex=complex))
    return U @ D @ U.conj().T

def random_projection(size, rank=None, orthogonal=True, complex=True):
    if rank is None:
        rank = np.random.randint(1, size+orthogonal)  # rank == n is always orthogonal (identity)
    else:
        rank = min(rank, size)

    # if orthogonal:
    #     # P^2 = P and P = P^H
    #     U = random_unitary(size)
    #     D = np.random.permutation([1]*rank + [0]*(size-rank))
    #     return U @ np.diag(D) @ U.conj().T
    # else:
    #     # P^2 = P, but almost never P = P^H
    #     A = random_square(size, complex=True)
    #     D = np.random.permutation([1]*rank + [0]*(size-rank))
    #     return A @ np.diag(D) @ np.linalg.pinv(A)

    # much faster for rank << size
    A = random_vec((size, rank), complex=complex)
    if orthogonal:
        B = A.conj().T
    else:
        B = random_vec((rank, size), complex=complex)
    return A @ np.linalg.pinv(B @ A) @ B

###################
### Polynomials ###
###################

def roots(coeffs):
    """ Solve a polynomial equation in one variable: $a_0x^n + a_1x^{n-1} + ... + a_{n-1}x + a_n = 0$."""
    def linear(a,b):
        """Solve a linear equation: ax + b = 0. """
        return -b/a

    def quadratic(a,b,c):
        """Solve a quadratic equation: ax^2 + bx + c = 0. """
        if abs(a) < 1e-10:
            return roots([b,c])
        d = (b**2 - 4*a*c)**(1/2)
        return (-b + d)/(2*a), (-b - d)/(2*a)

    def cubic(a,b,c,d):
        if abs(a) < 1e-10:
            return roots([b,c,d])
        """Solve a cubic equation: ax^3 + bx^2 + cx + d = 0. """
        del_0 = b**2 - 3*a*c
        del_1 = 2*(b**3) - 9*a*b*c + 27*(a**2)*d
        discriminant = del_1**2 - 4*(del_0**3)
        const = (del_1/2 + (1/2)*discriminant**(1/2))**(1/3)

        tmp1 = const
        tmp2 = const * np.exp(4/3*np.pi*1j)
        tmp3 = const * np.exp(8/3*np.pi*1j)
        pre = -1/(3*a)
        return pre*(b + tmp1 + del_0/tmp1), pre*(b + tmp2 + del_0/tmp2), pre*(b + tmp3 + del_0/tmp3)

    def quatric(a,b,c,d,e):
        """ Solve a quartic equation: ax^4 + bx^3 + cx^2 + dx + e = 0. """
        if abs(a) < 1e-10:
            return roots([b,c,d,e])
        p = c/a - 3*(b**2)/(8*(a**2))
        q = b**3/(8*(a**3)) - b*c/(2*(a**2)) + d/a
        delta_0 = c**2 - 3*b*d + 12*a*e
        delta_1 = 2*c**3 - 9*b*c*d + 27*(b**2)*e + 27*a*d**2 - 72*a*c*e
        discr = complex(delta_1**2 - 4*delta_0**3)
        Q = ((delta_1 + discr**(1/2))/2)**(1/3)
        S = 1/2 * (-2/3 * p + 1/(3*a) * (Q + delta_0/Q))**(1/2)
        pre = -1/(4*a) * b
        tmp0_1 = -S**2 - p/2
        tmp0_2 = q/(4*S)
        tmp1 = np.sqrt(tmp0_1 + tmp0_2)
        tmp2 = np.sqrt(tmp0_1 - tmp0_2)
        return pre - S - tmp1, pre - S + tmp1, pre + S - tmp2, pre + S + tmp2

    coeffs = list(map(complex, coeffs))
    if len(coeffs) < 2:
        raise ValueError("The degree of the polynomial must be at least 1.")
    elif len(coeffs) == 2:
        return linear(*coeffs)
    elif len(coeffs) == 3:
        return quadratic(*coeffs)
    elif len(coeffs) == 4:
        return cubic(*coeffs)
    elif len(coeffs) == 5:
        return quatric(*coeffs)
    else:
        while abs(coeffs[0]) < 1e-10:
            coeffs.pop(0)
            if len(coeffs) == 0:
                raise ValueError("The zero polynomial has roots everywhere!")
        return np.roots(coeffs)

if sage_loaded:
    def lagrange_multipliers(f, g, filter_real=False):
        """
        Solve a constrained optimization problem using the Lagrange multipliers.

        Parameters:
            f (sage.symbolic.expression.Expression): The objective function.
            g (list of sage.symbolic.expression.Expression): The equality constraints.

        Returns:
            list of dict: Solutions to the optimization problem.

        Example:
            >>> x, y, z = var('x y z')
            >>> f = (x-1)^2 + (y-1)^2 + (z-1)^2  # objective to minimize
            >>> g = x^4 + y^2 + z^2 - 1          # constraint (points need to lie on this surface)
            >>> lagrange_multipliers(f, g, filter_real=True)
        """
        g = g or []
        # multipliers
        if isinstance(g, Expression):
            g = [g]
        if len(g) == 0:
            lambdas = []
        else:
            lambdas = var(['lambda{}'.format(i) for i in range(len(g))])
        # Lagrange function
        L = f + sum(l * g for l, g in zip(lambdas, g))
        partials = list(L.gradient())
        # Solve the system
        solutions = solve(partials + g, L.variables(), solution_dict=True)
        # Filter out non-real solutions if desired
        sols = []
        for sol in solutions:
            for v in sol:
                sol[v] = sol[v].simplify_full()
            if filter_real:
                if not all(val.is_real() for val in sol.values()):
                    continue
            sol['f'] = f.subs(sol)
            sols.append(sol)
        return sols

    def polynomial_division(f, L, verbose=False):
        """ Division of polynomials. Returns the polynomials q and remainder r such that $f = r + \\sum_i q_i*g_i$ and either r = 0 or multideg(r) < multideg(g). If r == 0, we say "L divides f".

        Parameters:
            f (sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular)
            L (list[sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular])
            verbose (bool, optional): If True, print intermediate steps (default: False)

        Returns:
            q (list[sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular]): The quotients.
            r (sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular): The remainder.
        """
        if not isinstance(L, (list, tuple)):
            L = [L]
        q = [0] * len(L)
        r = 0
        p = f

        while p != 0:
            i = 0
            division_occurred = False

            while i < len(L) and not division_occurred:
                LTLi = L[i].lt()
                LTp = p.lt()
                if verbose:
                    print(i, f"LT(L_i) = {LTLi} divides LT(p) = {LTp}? ", LTLi.divides(LTp))
                if LTLi.divides(LTp):
                    nqt = (LTp / LTLi).numerator()
                    q[i] += nqt
                    p -= nqt * L[i]
                    division_occurred = True
                    if verbose:
                        print(f"Set q_i = {q[i]} and p = {p}")
                else:
                    i += 1

            if not division_occurred:
                LTp = p.lt()
                r += LTp
                p -= LTp
                if verbose:
                    print(f"No divison occurred -> Set r = r + LT(p) = {r} and p = p - LT(p) = {p}")

        return q, r

    def s_polynomial(f, g, verbose=False):
        """
        Compute the S-polynomial of two polynomials.

        Parameters:
            f (sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular)
            g (sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular)

        Returns:
            sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular: The S-polynomial.
        """
        flm, glm = f.lm(), g.lm()
        lcmfg = lcm(flm, glm)
        if verbose:
            print(f"lcm({flm}, {glm}) = {lcmfg}")
        res = lcmfg * (f / f.lt() - g / g.lt())
        assert res.denominator() == 1, "Not a polynomial"
        return res.numerator()

    def Buchberger(F, verbose=False):
        """
        Compute a Gröbner basis using the Buchberger algorithm.

        Parameters:
            F (list[sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular]): The list of polynomials.
            verbose (bool, optional): If True, print intermediate steps.

        Returns:
            list[sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular]: The Gröbner basis.
        """
        G = F.copy()
        G_ = []
        while G != G_:
            G_ = G.copy()
            for p, q in combinations(G, 2):
                sp = s_polynomial(p, q)
                _, r = polynomial_division(sp, G)
                if r != 0:
                    G.append(r)
                    if verbose:
                        print(f"{p} and {q} give {sp} which reduces to {r}")
                        print(f"G is now {G}")
        return G

    def is_groebner_basis(G, verbose=False):
        """ Use Buchberger's criterion to check whether G is a Gröbner basis. """
        for p, q in combinations(G, 2):
            sp = s_polynomial(p, q)
            _, r = polynomial_division(sp, G)
            if r != 0:
                if verbose:
                    print(f"S({p}, {q}) = {sp} reduces to {r}")
                return False
        return True

    def is_minimal_groebner_basis(G, verbose=False):
        # 1. G is a Gröbner basis
        if not is_groebner_basis(G, verbose):
            return False
        # 2. All leading coefficients are 1
        for p in G:
            if p.lc() != 1:
                if verbose:
                    print(f"Leading coefficient of {p} is not 1")
                return False
        # 3. No leading monomial is divisible by the leading monomial of another polynomial
        for p, q in combinations(G, 2):
            if p.lt().divides(q.lt()):
                if verbose:
                    print(f"LT({p}) = {p.lt()} divides LT({q}) = {q.lt()}")
                return False
        return True

    def is_reduced_groebner_basis(G, verbose=False):
        # 1. G is a minimal Gröbner basis
        if not is_minimal_groebner_basis(G, verbose):
            return False
        # 2. No monomial of any polynomial in G is divisible by the leading monomial of another polynomial
        for p, q in combinations(G, 2):
            for m in p.monomials():
                if q.lt().divides(m):
                    if verbose:
                        print(f"LT({q}) = {q.lt()} divides {m}, which is a monomial of {p}")
                    return False
        return True

    def elimination_ideal(I, variables):
        """
        Compute the elimination ideal of a given ideal.

        Parameters:
            I (sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal): The ideal.
            variables (list[sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular]): The variables to eliminate.

        Returns:
            sage.rings.polynomial.multi_polynomial_ideal.MPolynomialIdeal: The elimination ideal.
        """
        if not isinstance(variables, (list, tuple)):
            variables = [variables]
        R = I.ring()
        return R.ideal([g for g in I.groebner_basis() if all(x not in g.variables() for x in variables)])

    def generate_names(names=None, s=None, k=None):
        """ Give either `names`, or `s` and `k`. Returns as list. """
        if names is None:
            names = [f'{s}{i}' for i in range(k)]
        else:
            if isinstance(names, str):
                names = names.split(', ')
                if len(names) == 1:
                    names = names[0].split(' ')
                    if len(names) == 1:
                        names = names[0].split(',')
            if k is not None:
                assert len(names) == k, f"Expected {k} names for {s}, but got {len(names)}: {names}"
        return names

    def implicitization(r, m, tnames=None, xnames=None):
        R = PolynomialRing(QQ, ['t{}'.format(i) for i in range(m)], order='lex')
        tmp = r(*R.gens())
        n = len(tmp)
        tnames = generate_names(tnames, 't', m)
        xnames = generate_names(xnames, 'x', n)

        R = PolynomialRing(QQ, ['λ'] + tnames + xnames, order='lex')
        l = R.gens()[0]
        t = R.gens()[1:m+1]
        x = R.gens()[m+1:]
        r = r(*t)
        p = [gi.numerator() for gi in r]
        q = [gi.denominator() for gi in r]
        if all(qi == 1 for qi in q):
            gens = [xi - pi for xi, pi in zip(x, p)]
            toeliminate = list(t)
        else:
            gens = [qi*xi - pi for xi, pi, qi in zip(x, p, q)] + [1-l*prod(q)]
            toeliminate = [l] + list(t)
        I = R.ideal(gens)
        return [I] + [I.elimination_ideal(toeliminate[:i+1]) for i in range(len(toeliminate))]

    def reduction(f):
        """ Return the reduction of f. This is the same as `R.ideal(f).radical().gens()[0]`. """
        return f / gcd(f, *f.gradient())

    def random_polynomial(variables, n_terms=10, max_degree=2, seed=None):
        """Generate a random polynomial with integer coefficients."""
        import random
        from random import randint

        seed = randint(0, 1000) if seed is None else seed
        random.seed(int(seed))
        # monomials = [prod([var^randint(0, q) for var in variables]) for _ in range(randint(1, n))]
        monomials = [prod([var^randint(0, max_degree) for var in variables]) for _ in range(1, n_terms)]
        return sum([randint(-100, 100)*m for m in monomials])

#####################
### Number theory ###
#####################

def _two_arguments(f, *a):
    if len(a) == 2:
        return a
    elif len(a) == 3:
        return a[0], f(a[1], a[2])
    elif len(a) > 3:
        mid = len(a) // 2
        return f(*a[:mid]), f(*a[mid:])
    elif len(a) == 1:
        if hasattr(a[0], '__iter__'):
            return _two_arguments(f, *a[0])
        return a[0], None
    raise ValueError("No arguments given.")

def gcd(*a):
    """ Greatest common divisor. """
    a, b = _two_arguments(gcd, a)
    if hasattr(a, 'gcd'):
        return a.gcd(b)
    if not hasattr(a, '__bool__'):
        raise ValueError(f"Can't calculate gcd of {a} and {b}.")
    while b:
        a, b = b, a % b
    return a

def is_coprime(*a):
    """ Test if `a` and `b` are coprime. """
    return gcd(*a) == 1

if not sage_loaded:

    def is_prime(n, alpha=1e-20): # only up to 2^54 -> alpha < 1e-16.26 (-> 55 iterations; < 1e-20 is 67 iterations)
        """Miller-Rabin test for primality."""
        if n == 1 or n == 4:
            return False
        if n == 2 or n == 3:
            return True

        def getKM(n):
            k = 0
            while n % 2 == 0:
                k += 1
                n /= 2
            return k,int(n)

        p = 1
        while p > alpha:
            a = randint(2,n-2)
            if gcd(a,n) != 1:
                # print(n,"is not prime (1)", f"gcd({a},{n}) = {gcd(a,n)}")
                return False
            k,m = getKM(n-1)
            b = pow(a, m, n)
            if b == 1:
                p *= 1/2
                continue
            for i in range(1,k+1):
                b_new = pow(b,2,n)
                # first appearance of b == 1 is enough
                if b_new == 1:
                    break
                b = b_new
                if i == k:
                    # print(n,"is not prime (2)")
                    return False
            if gcd(b+1,n) == 1 or gcd(b+1,n) == n:
                p *= 1/2
            else:
                # print(n,"is not prime (3)")
                return False

        # print("%d is prime with alpha=%E (if Carmichael number: alpha=%f)" % (n, p, (3/4)**log(p,1/2)))
        return True

    def prime_factors(n):
        """Simple brute-force algorithm to find prime factors"""
        factors = []
        # remove all factors of 2 first
        while n % 2 == 0:
            factors.append(2)
            n //= 2
        i = 3
        while i * i <= n:
            if n % i:
                i += 2
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors

    def euler_phi(n):
        """ Euler's totient function. """
        return round(n * np.prod([1 - 1/p for p in prime_factors(n)]))
        # return sum([1 for k in range(1, n) if is_coprime(n, k)])  # ~100x slower

    def lcm(*a):
        """ Least common multiple. """
        a, b = _two_arguments(lcm, a)
        if b is None:
            return a
        if hasattr(a, 'lcm'):
            return a.lcm(b)
        return a * b // gcd(a, b)

def closest_prime_factors_to(n, m):
    """Find the set of prime factors of n with product closest to m."""
    pf = prime_factors(n)

    min_diff = float("inf")
    min_combo = None
    for c in powerset(pf):
        diff = abs(m - np.prod(c))
        if diff < min_diff:
            min_diff = diff
            min_combo = c
    return min_combo

def int_sqrt(n):
    """For integer $n$, find the integer $a$ closest to $\\sqrt{n}$, such that $n/a$ is also an integer."""
    if n == 1 or n == 0:
        return n
    return int(np.prod(closest_prime_factors_to(n, sqrt(n))))

def dlog(x,g,n):
    """ Discrete logarithm, using the baby-step giant-step algorithm.

    Parameters
        x (int): The number to find the logarithm of.
        g (int): The base.
        n (int): The modulus.

    Returns
        int: The discrete logarithm of `x` to the base `g` modulo `n`.
    """
    w = ceil(sqrt(euler_phi(n)))
    baby = []
    for j in range(w+1):
        baby.append(x*(g**j) % n)
#     print(baby)
    for i in range(1, w+1):
        giant = pow(g, i*w, n)
#         print(i, giant)
        if giant in baby:
            return i * w-baby.index(giant) % n

    raise ValueError(f"Couldn't find discrete logarithm of {x} to the base {g} modulo {n}.")

def is_carmichael(n):
    """ Test if `n` is a Carmichael number. """
    if is_prime(n): # are neither even nor prime
        return False
    for a in range(2,n):
        if gcd(n,a) == 1 and pow(a,n-1,n) != 1:
            return False
    return True

def carmichael_numbers(to):
    for n in range(3*5*7, to, 2): # have at minimum three prime factors and not even
        if is_carmichael(n):
            yield n

######################
### Binary strings ###
######################

def binFrac_i(j, i):
    return int(j % (1/2**(i-1)) != j % (1/2**i))
    #return int(np.ceil((j % (1/2**(i-1))) - (j % (1/2**i))))

def binFrac(j, prec=20):
    return "." + "".join([str(binFrac_i(j,i)) for i in range(1,prec+1)])

def binstr_from_float(f, r=None, complement=False):
    """
    Convert a float `f` to a binary string with `r` bits after the comma.
    If `r` is None, the number of bits is chosen such that the float is
    represented exactly.

    Parameters
        f (float): The float to convert.
        r (int), optional: The number of bits after the comma. The default is None.
        complement (bool), optional: If True and `f < 0`, count the fraction "backwards" (e.g. -0.125 == '-.111').

    Returns
        str: The binary string representing `f`.
    """
    negative = f < 0
    if negative:
        f = -f # make it easier to handle the minus sign in the end
        if r is not None and r > 0 and abs(f) < 1/2**(r+1):
            return '.' + '0'*r
        if complement:
            # Translate the fraction to the corresponding complement, e.g. -0.125 => -0.875
            # alternatively, we could also flip all bits in `frac_part` below and add 1
            frac = f - int(f)
            if frac > 0:
                f = int(f) - frac + 1  # -1 if f was negative

    i = 0 # number of bits in the fraction part
    while int(f) != f:
        if r is not None and i >= r:
            f = int(np.round(f))
            break
        f *= 2
        i += 1
    f = int(f) # there should be no fractional part left

    # We use complement only for the fraction, not for the integer part
    # # If `f` is negative, the positive number modulus `2**k` is returned,
    # # where `k` is the smallest integer such that `2**k > -f`.
    # if f < 0:
    #     k = 0
    #     while -f > 2**(k-1):
    #         k += 1
    #     f = 2**k + f

    # integer part
    as_str = str(bin(f))[2:] # this adds a leading '-' sign for negative numbers
    sign = '-' if negative else ''
    # print(f, i, sign, as_str)
    if i == 0: # no fraction part
        if r is None or r <= 0: # ==> i == 0
            return sign + as_str
        if as_str == '0':
            return sign + '.' + '0'*r
        return sign + as_str + '.' + '0'*r
    int_part = sign + as_str[:-i]

    # fraction part
    frac_part = '0'*(i-len(as_str)) + as_str[-i:]
    # print(int_part, frac_part)
    if r is None:
       return int_part + '.' + frac_part
    return int_part + '.' + frac_part[:r] + '0'*(r-len(frac_part[:r]))

def float_from_binstr(s, complement=False):
    """ Convert a binary string to a float.

    Parameters
        s (str): The binary string.
        complement (bool, optional): If True, interpret the fraction part as the complement of the binary representation. Defaults to False.

    Returns
        float: The float represented by the binary string.
    """

    negative = s[0] == '-'
    if negative:
        s = s[1:]
    s = s.split('.')

    pre = 0
    frac = 0
    if len(s[0]) > 0:
        pre = int(s[0], 2)
    if len(s) > 1 and len(s[1]) > 0:
        if negative and complement:
            # flip all bits and add 1
            s[1] = ''.join(['1' if x == '0' else '0' for x in s[1]])
            frac = int(s[1], 2) + 1
        else:
            frac = int(s[1], 2)
        frac /= 2.**len(s[1])
    return float(pre + frac) * (-1 if negative else 1)

def binstr_from_int(n, places=0):
    if places > 0:
        if n < 0:
            return '-'+binstr_from_int(-n, places)
        res = f"{n:0{places}b}"
        if len(res) > places:
            raise ValueError(f"Integer {n} can't be represented in {places} bits")
        return res
    return f"{n:b}"

def int_from_binstr(s):
    return int(float_from_binstr(s))

def bincoll_from_binstr(s):
    return [int(x) for x in s]

def binstr_from_bincoll(l):
    return "".join([str(x) for x in l])

def int_from_bincoll(l):
    #return sum([2**i*v_i for i,v_i in enumerate(reversed(l))])
    return int_from_binstr(binstr_from_bincoll(l))

def bincoll_from_int(n, places=0):
    return bincoll_from_binstr(binstr_from_int(n, places))

############
### Misc ###
############

### useful

def softmax(a, beta=1):
    a = np.exp(beta*a)
    Z = np.sum(a)
    return a / Z

def choice(a, size=None, replace=True, p=None):
    if p is not None:
        if np.abs(np.sum(p) - 1) > sys.float_info.epsilon:
            p = normalize(p, p=1)

    if hasattr(a, '__len__'):
        n = len(a)
        idx = np.random.choice(n, size=size, replace=replace, p=p)
        return np.array(a)[idx]
    else:
        return np.random.choice(a, size=size, replace=replace, p=p)

if not sage_loaded:
    # https://docs.python.org/3/library/itertools.html
    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

### special numbers

Phi = (1 + np.sqrt(5))/2
def Fibonacci(n):
    Psi = 1 - Phi
    return int(np.round((Phi**n - Psi**n)/(Phi - Psi))) # /np.sqrt(5)

def calc_pi1(prec=100):
    """ Calculate pi using the Gauss-Legendre algorithm. """
    from decimal import Decimal, getcontext
    getcontext().prec = prec # int(np.e*2**N-2)
    N = int(np.log2(prec + 2))
    a = Decimal(1)
    b = Decimal(1)/Decimal(2).sqrt()
    t = Decimal(1)/Decimal(4)
    p = Decimal(1)
    for n in range(N):
        a_next = (a+b)/2
        b_next = (a*b).sqrt()
        t_next = t - p*(a-a_next)**2
        p_next = 2*p
        a = a_next
        b = b_next
        t = t_next
        p = p_next
    return (a+b)**2/(4*t)

def calc_pi2(prec=100):
    """ Calculate pi using the Chudnovsky algorithm. """
    from decimal import Decimal, getcontext
    getcontext().prec = prec  # int(14*N)
    N = int(prec/14)
    r = Decimal(0)
    for n in range(N):
        r_i = Decimal(factorial(6*n)*(13591409+545140134*n))/Decimal(factorial(3*n)*(factorial(n)*(-640320)**n)**3)
        r += r_i
    return Decimal(4270934400)/(Decimal(10005).sqrt()*r)

def calc_pi3(prec=100):
    """ Calculate pi using the Ramanujan algorithm. """
    from decimal import Decimal, getcontext
    getcontext().prec = prec  # int(8*N)
    N = int(prec/7)
    r = Decimal(0)
    for n in range(N):
        r_i = Decimal(factorial(4*n)*(1103+26390*n))/Decimal((factorial(n)*396**n)**4)
        r += r_i
    return Decimal(9801)/(Decimal(8).sqrt()*r)

def calc_pi4(prec=100):
    """ Calculate pi using the Bailey-Borwein-Plouffe algorithm. """
    # return BBP_formula(1, 16, 8, [4, 0, 0, -2, -1, -1])
    from decimal import Decimal, getcontext
    getcontext().prec = prec  # int(1.25*N+2.5)
    N = int((prec-2.5)/1.2)
    r = Decimal(0)
    for n in range(N):
        r_i = Decimal(1)/Decimal(16**n)*(Decimal(4)/(8*n+1)-Decimal(2)/(8*n+4)-Decimal(1)/(8*n+5)-Decimal(1)/(8*n+6))
        r += r_i
    return r

def BBP_formula(s, b, m, a, N=100, verbose=False):
    """ Calculate the sum of the series $sum_{k=0}^N 1/(b^k) (sum_{j=1}^m a_j/(mk+j)^s)$.
    This is a general form to calculate irrational numbers, e.g.
    - `np.pi == BBP_formula(1, 16, 8, [4, 0, 0, -2, -1, -1])`
    - `np.log(a) - np.log(a-1) == 1/2*BBP_formula(1, a, 1, [1])`

    See also https://en.wikipedia.org/wiki/Bailey–Borwein–Plouffe_formula
    """
    def P(k, _):
        return 1/(b**k)*sum([a[j]/(m*k+j+1)**s for j in range(len(a))])

    return series(P, start_value=0, start_index=-1, max_iter=N+1, verbose=verbose)

def log_(x, base=np.e):
    """
    Calculating the logarithm by foot: Use that $\\ln(1+x) = \\int_0^x \\frac 1{1+t} dt = \\sum_{n=0}^\\infty (-1)^n \\frac{x^{n+1}}{n+1}$. 
    However, this series converges only for $|x| < 1$ and quickly only for $|x| < 1/2$. So, divide x by e until 0.5 < x' < 1.5.
    Then use that ln(e^k*x') = k + ln(x').
    Speedup by using 2 instead of e (easier divison / multiplication), and that $ln(1+x) - ln(1-x)$ only contains terms with even powers (faster convergence).
    """
    if base != np.e:
        return log_(x)/log_(base)
    if x < 0:
        return np.nan
    if x == 0:
        return -np.inf
    k = 0
    ln2 = np.log(2)
    if x > 0.5:
        while np.abs(x - 1) >= 0.5:
            x /= 2
            k += 1
    else:
        while np.abs(x - 1) >= 0.5:
            x *= 2
            k -= 1
    # x_ = x - 1
    # lnx_ = series(lambda n, _: (-1)**n * x_**(n+1)/(n+1), start_index=-1)
    x_ = (x-1)/(x+1)
    lnx_ = series(lambda n, _: 2/(2*n+1) * x_**(2*n+1), start_index=-1)
    return k*ln2 + lnx_

log_table, log_table_keys = None, None
def log_2(x, base=np.e):
    """
    Calculating the logarithm using Feynman's algorithm. Performance gets close (~ factor 5) to np.log.
    """
    if base != np.e:
        return log_2(x)/log_2(base)
    if x < 0:
        return np.nan
    if x == 0:
        return -np.inf

    global log_table, log_table_keys
    if log_table is None:
        log_table, log_table_keys = [], []
        # build the table
        for k in range(52+1):  # 52-bit precision in 64-bit float
            k2k = 1 + 2**(-k)
            log_table_keys.append(k2k)
            log2k = np.log(k2k)
            log_table.append(log2k)

    # bring x in range 1 <= x < 2
    k = 0
    if x > 1:
        while x >= 2:
            x /= 2
            k += 1
    else:
        while x < 1:
            x *= 2
            k -= 1

    # calculate ln(x) by composing x as factors 1 + 2^{-k}
    x_ = 1
    lnx_ = 0
    for j in range(1,53):
        x__ = x_ * log_table_keys[j]
        if x__ < x:
            x_ = x__
            lnx_ += log_table[j]

    return k*log_table[0] + lnx_

### more

def sqrt_brain_compatible(x, correction_term=False, n_max = 20):
    """ Nice way to calculate approximate square roots in the head:
        1. Find the largest integer n, such that n² < x
        2. sqrt(x) ≈ n + (x-n²)/(2n+1) + 1/(6(2n+1))
    The last term is more negligible the larger x.
        """
    n_sq_table = [n**2 for n in range(1,n_max)]
    if x <= 1:
       raise ValueError("Are you kidding me")
    if x > n_sq_table[-1]:
       raise ValueError("Use a proper function for this, please.")
    for n,n2 in enumerate(n_sq_table):
       if n2 > x:
           if correction_term:
               return n + (x - n_sq_table[n-1])/(2*n + 1) + 1/(6*(2*n+1))
           else:
               return n + (x - n_sq_table[n-1])/(2*n + 1)

# Reduction of the halting problem to the equivalence problem, i.e. show that the latter is at least as hard as the former.
# Give an algorithm `H` and input `x` to decide whether `H(x)` halts. This function returns two functions `f1, f2`. Use your
# implementation `equiv` solving the equivalence problem to solve the halting problem, e.g. `equiv(*equiv_from_halt(H, x))`.
def equiv_from_halt(H, x):
    def f1(y):
        if y == x:
            return H(x)
        return True

    def f2(y):
        return True

    return f1, f2

def Hutchinson_trace(A, n=1000):
    """ Terribly inefficient, but theoretically interesting. """
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    zs = np.random.normal(0, 1, size=(A.shape[0], n))
    return np.mean([z.T @ A @ z for z in zs.T])


### Tests

def test_mathlib_all():
    tests = [
        _test_series,
        _test_sequence,
        _test_normalize,
        _test_rad,
        _test_deg,
        _test_is_symmetric,
        _test_random_symmetric,
        _test_is_orthogonal,
        _test_random_orthogonal,
        _test_is_hermitian,
        _test_random_hermitian,
        _test_is_unitary,
        _test_random_unitary,
        _test_is_psd,
        _test_random_psd,
        _test_is_normal,
        _test_random_normal,
        _test_is_projection,
        _test_random_projection,
        _test_matexp,
        _test_matlog,
        _test_immanant,
        _test_roots,
        _test_gcd,
        _test_is_coprime,
        _test_closest_prime_factors_to,
        _test_int_sqrt,
        _test_dlog,
        _test_is_carmichael,
        _test_binFrac,
        _test_binstr_from_float,
        _test_float_from_binstr,
        _test_binstr_from_int,
        _test_int_from_binstr,
        _test_bincoll_from_binstr,
        _test_binstr_from_bincoll,
        _test_int_from_bincoll,
        _test_bincoll_from_int,
        _test_softmax,
        _test_Fibonacci,
        _test_calc_pi,
        _test_log_
    ]
    if sage_loaded:
        tests += [
            _test_lagrange_multipliers,
            _test_polynomial_division,
            _test_s_polynomial,
            _test_elimination_ideal,
            _test_implicitization,
            _test_reduction
        ]
    else:
        tests += [
            _test_SO,
            _test_su,
            _test_SU,
            _test_is_prime,
            _test_prime_factors,
            _test_euler_phi,
            _test_lcm,
            _test_powerset
        ]

    for test in tests:
        print("Running", test.__name__, "... ", end="")
        test()
        print("Test succeeded!")

def _test_series():
    res = series(lambda n, _: 1/factorial(2*n), 1) + series(lambda n, _: 1/factorial(2*n + 1), 1)
    assert np.isclose(res, np.e)

    res = series(lambda n, _: 1/(2**n))
    assert np.isclose(res, 1)

    # pauli X
    A0 = np.array([[0, 1.], [1., 0]])
    a = np.eye(A0.shape[0]) + series(lambda n, A: A @ A0 / n, start_value=A0, start_index=1)
    expected = np.array([[np.cosh(1), np.sinh(1)], [np.sinh(1), np.cosh(1)]])
    assert np.allclose(a, expected)

    # pauli Y
    A0 = np.array([[0, -1j], [1j, 0]])
    a = np.eye(A0.shape[0]) + series(lambda n, A: A @ A0 / n, start_value=A0, start_index=1)
    expected = np.array([[np.cosh(1), -1j*np.sinh(1)], [1j*np.sinh(1), np.cosh(1)]])
    assert np.allclose(a, expected)

    # pauli Z
    A0 = np.array([[1., 0], [0, -1.]])
    a = np.eye(A0.shape[0]) + series(lambda n, A: A @ A0 / n, start_value=A0, start_index=1)
    expected = np.array([[np.e, 0], [0, 1/np.e]])
    assert np.allclose(a, expected)

def _test_sequence():
    # for any converging series, the sequence should converge to 0
    res = sequence(lambda n, _: 1/factorial(2*n), 1)
    assert np.isclose(res, 0)

    # a nice fractal to test the matrix version
    x,y = np.meshgrid(np.linspace(-0.7,1.7,200), np.linspace(-1.1,1.1,200))
    warnings.filterwarnings("ignore")  # ignore overflow warnings
    res = sequence(lambda i,x: (0.3+1j)*x*(1-x), start_value=x+1j*y, max_iter=200)
    warnings.filterwarnings("default")
    res[np.isnan(res)] = np.inf
    assert np.allclose(np.mean(np.isinf(res)), 0.78815)

def _test_normalize():
    a = random_vec(randint(2,20), complex=True)
    b = normalize(a)
    assert np.isclose(np.linalg.norm(b), 1)
    a = np.array(3 - 4j)
    assert np.isclose(normalize(a), a/5)

def _test_rad():
    assert rad(180) == np.pi
    assert rad(0) == 0

def _test_deg():
    assert deg(np.pi) == 180
    assert deg(0) == 0

def _test_is_symmetric():
    a = random_square(randint(2,20), complex=False)
    b = a + a.T
    assert is_symmetric(b)

    c = a + 1
    assert not is_symmetric(c)

def _test_random_symmetric():
    a = random_symmetric(randint(2,20))
    assert is_symmetric(a)

def _test_is_orthogonal():
    a, b = np.random.rand(2)
    a, b = normalize([a, b])
    a = np.array([
        [a, b],
        [-b, a]
    ])
    assert is_orthogonal(a)

    c = a + 1
    assert not is_orthogonal(c)

def _test_random_orthogonal():
    a = random_orthogonal(randint(2,20))
    assert is_orthogonal(a)

def _test_is_hermitian():
    a = random_square(randint(2,20), complex=True)
    b = a + a.conj().T
    assert is_hermitian(b)
    assert is_antihermitian(1j*b)
    c = a + 1
    assert not is_hermitian(c)
    assert not is_antihermitian(1j*c)

def _test_random_hermitian():
    a = random_hermitian(5)
    assert is_hermitian(a)

def _test_is_unitary():
    assert is_unitary(np.eye(randint(2,20)))

    a, b = random_vec(2, complex=True)
    a, b = normalize([a, b])
    phi = np.random.rand()*2*np.pi
    a = np.array([
        [a, b],
        [-np.exp(1j*phi)*b.conjugate(), np.exp(1j*phi)*a.conjugate()]
    ])
    assert is_unitary(a)

    A = random_square(randint(2,20), complex=True)
    J = matsqrt(A.T.conj() @ A)
    U = A @ np.linalg.pinv(J)  # polar decomposition
    assert is_unitary(U)

    c = a + 1
    assert not is_unitary(c)

def _test_random_unitary():
    a = random_unitary(randint(2,20))
    assert is_unitary(a)

def _test_is_psd():
    # A @ A^\dagger => PSD
    a = random_square(randint(2,20), complex=True)
    a = a @ a.conj().T
    assert is_psd(a)

    # unitarily diagonalizable (= normal) + positive eigenvalues <=> PSD
    U = random_unitary(5)
    p = np.random.rand(5)
    a = U @ np.diag(p) @ U.conj().T
    assert is_psd(a)

    # sum(p) can't be larger than 5 here, so make the trace negative to guarantee negative eigenvalues
    b = a - 5
    assert not is_psd(b)

def _test_random_psd():
    a = random_psd(randint(2,20))
    assert is_psd(a)

def _test_is_normal():
    H = random_hermitian(randint(2,20))
    assert is_normal(H)
    U = random_unitary(randint(2,20))
    assert is_normal(U)
    P = random_psd(randint(2,20))
    assert is_normal(P)
    A = random_square(randint(2,20))
    assert not is_normal(A)  # a random matrix is not normal

def _test_random_normal():
    N = random_normal(randint(2,20))
    assert is_normal(N)
    assert commute(N, N.T.conj())

def _test_is_projection():
    # orthogonal projection
    P = np.array([[1, 0], [0, 0]])
    assert is_projection(P)
    assert is_projection_orthogonal(P)
    # oblique projection
    P = np.array([[1, 0], [1, 0]])
    assert is_projection(P)
    assert not is_projection_orthogonal(P)
    P = np.array([[0, 0], [np.random.normal(), 1]])
    assert is_projection(P)
    assert not is_projection_orthogonal(P)
    # not a projection
    P = np.array([[1, 1], [1, 1]])  # evs 0 and 2
    assert not is_projection(P)
    assert not is_projection_orthogonal(P)

def _test_random_projection():
    n = 15
    P = random_projection(n, orthogonal=True)
    assert is_projection(P)
    assert is_projection_orthogonal(P)
    assert not is_orthogonal(P)  # just to be clear on that

    rank = randint(2,n)
    P = random_projection(n, rank=rank, orthogonal=True)
    assert is_projection(P)
    assert is_projection_orthogonal(P)
    assert np.linalg.matrix_rank(P) == rank

    P = random_projection(n, orthogonal=False)
    assert is_projection(P)
    assert not is_projection_orthogonal(P)  # can technically still be orthogonal

    P = random_projection(n, rank=rank, orthogonal=False)
    assert is_projection(P)
    assert not is_projection_orthogonal(P)
    assert np.linalg.matrix_rank(P) == rank

def _test_matexp():
    a = random_square(randint(2,20), complex=True)
    # check if det(matexp(A)) == exp(trace(A))
    assert np.isclose(np.linalg.det(matexp(a)), np.exp(np.trace(a)))

def _test_matlog():
    alpha = np.random.rand()*2*np.pi - np.pi
    A = np.array([[np.cos(alpha), -np.sin(alpha)],[np.sin(alpha), np.cos(alpha)]])
    assert np.allclose(matlog(A), alpha*np.array([[0, -1],[1, 0]])), f"Error for alpha = {alpha}! {matlog(A)} != {alpha*np.array([[0, -1],[1, 0]])}"

    U = random_unitary(randint(2,20))  # any unitary is exp(-iH) for some hermitian H
    assert is_antihermitian(matlog(U))

def _test_immanant():
    A = np.array([[1, 2], [3, 4]])
    assert np.isclose(determinant(A), -2), f"{determinant(A)} ≠ -2"
    assert np.isclose(permanent(A), 10), f"{permanent(A)} ≠ 10"
    A = random_square(randint(2,6))
    assert np.isclose(determinant(A), np.linalg.det(A)), f"{determinant(A)} ≠ {np.linalg.det(A)}"

def _test_roots():
    # Test cases
    assert np.allclose(roots([1,0,-1]), (1.0, -1.0))
    assert np.allclose(roots([1,0,1]), (1j, -1j))
    assert np.allclose(roots([1, -2, -11, 12]), [-3, 1, 4])
    assert np.allclose(roots([1, -7, 5, 31, -30]), [-2, 1, 3, 5])
    assert np.allclose(roots([0, -1, 2, 3]), (-1, 3))

    for degree in range(1, 6):
        coeffs = random_vec(degree+1, (-10, 10), complex=True)
        assert np.allclose(np.polyval(coeffs, roots(coeffs)), 0), f"{coeffs}: {roots(coeffs)}"

    p = Polynomial([-4, 3.5, 2.5, 0, 0])
    for c in p.roots:
        assert np.isclose(p(c), 0)

def _test_SO():
    n = 4
    SOn = SO(n)

    # check the number of generators
    n_expected = n*(n-1)//2
    assert len(SOn) == n_expected, f"Number of generators is {len(SOn)}, but should be {n_expected}!"

    # check if all generators are orthogonal
    for i, A in enumerate(SOn):
        random_angle = np.random.randn()
        assert is_orthogonal(A(random_angle)), f"Generator {i} is not orthogonal! ({random_angle})"

    # check if all generators are determinant 1
    for i, A in enumerate(SOn):
        random_angle = np.random.randn()
        assert np.isclose(np.linalg.det(A(random_angle)), 1), f"Generator {i} does not have determinant 1! ({random_angle})"

def _test_su():
    n = randint(2**1, 2**3)
    sun = su(n)

    # check the number of generators
    n_expected = n**2-1
    assert len(sun) == n_expected, f"Number of generators is {len(sun)}, but should be {n_expected}!"

    # check if no two generators are the same
    for i, (A,B) in enumerate(combinations(sun,2)):
        assert not np.allclose(A, B), f"Pair {i} is not different!"

    # check if all generators are traceless
    for i, A in enumerate(sun):
        assert np.isclose(np.trace(A), 0), f"Generator {i} is not traceless!"

    # check if all generators are Hermitian
    for i, A in enumerate(sun):
        assert is_hermitian(A), f"Generator {i} is not Hermitian!"

    # check if all generators are pairwise orthogonal
    for i, (A,B) in enumerate(combinations(sun,2)):
        assert np.allclose(np.trace(A.conj().T @ B), 0), f"Pair {i} is not orthogonal!"

    # check normalization
    su_n_norm = su(n, normalize=True)
    for i, A in enumerate(su_n_norm):
        assert np.isclose(np.linalg.norm(A), 1), f"Generator {i} does not have norm 1!"

    # check sparse representation
    sun_sp = su(n, sparse=True)

    # check the generators are the same
    for i, (A,B) in enumerate(zip(sun, sun_sp)):
        assert np.allclose(A, B.todense()), f"Pair {i} is not the same!"

def _test_SU():
    n = 4
    SUn = SU(n)

    # check the number of generators
    n_expected = n**2-1
    assert len(SUn) == n_expected, f"Number of generators is {len(SUn)}, but should be {n_expected}!"

    # check if all generators are unitary
    for i, A in enumerate(SUn):
        random_angle = np.random.randn()
        assert is_unitary(A(random_angle)), f"Generator {i} is not unitary! ({random_angle})"

    # check if all generators have determinant 1
    warnings.filterwarnings("ignore")  # ignore numpy warnings (bug)
    for i, A in enumerate(SUn):
        random_angle = np.random.randn()
        assert np.isclose(np.linalg.det(A(random_angle)), 1), f"Generator {i} is not in SU({n})! ({random_angle})"
    warnings.filterwarnings("default")

    # check if no two generators are the same
    for i, (A,B) in enumerate(combinations(SUn,2)):
        random_angle = np.random.randn()
        assert not np.allclose(A(random_angle), B(random_angle)), f"Pair {i} is not different! ({random_angle})"

def _test_lagrange_multipliers():
    tol = 1e-10
    def assert_dict_close(a, b):
        for k in a.keys():
            k_ = str(k)
            assert k_ in b, f"Key {k} not in b!"
            assert abs(a[k]-b[k_]) < tol, f"Value {a[k]} != {b[k_]}!"

    x, y, z = var('x y z')
    f = (x-1)**2 + (y-1)**2 + (z-1)**2   # objective to minimize
    g = x**2 + y**2 + z**2 - 3           # constraint (points need to lie on this surface)

    sols = lagrange_multipliers(f, g)
    assert len(sols) == 2
    assert abs(sols[0]['f']) < tol or abs(sols[1]['f']) < tol
    assert_dict_close(sols[0], {'lambda0': -2, 'x': -1, 'y': -1, 'z': -1, 'f': 12}), sols
    assert_dict_close(sols[1], {'lambda0': 0, 'x': 1, 'y': 1, 'z': 1, 'f': 0}), sols

    g2 = x*y - 1   # another constraint
    sols = lagrange_multipliers(f, [g, g2])
    assert len(sols) == 4
    assert_dict_close(sols[0], {'lambda0': -2, 'lambda1': 0, 'x': -1, 'y': -1, 'z': -1, 'f': 12}), sols
    assert_dict_close(sols[1], {'lambda0': 0, 'lambda1': -4, 'x': -1, 'y': -1, 'z': 1, 'f': 8}), sols
    assert_dict_close(sols[2], {'lambda0': 0, 'lambda1': 0, 'x': 1, 'y': 1, 'z': 1, 'f': 0}), sols
    assert_dict_close(sols[3], {'lambda0': -2, 'lambda1': 4, 'x': 1, 'y': 1, 'z': -1, 'f': 4}), sols

def _test_polynomial_division():
    R = PolynomialRing(QQ, 'x')
    x, = R.gens()
    f = x**2 - 1
    g = x - 1
    q, r = polynomial_division(f, g)
    assert q == [x + 1] and r == 0, f"q = {q}, r = {r}"

    R = PolynomialRing(QQ, 'x, y')
    x,y = R.gens()
    f = x**2*y + x*y**2 + y**2
    f1 = x*y - 1
    f2 = y**2 - 1
    q,r = polynomial_division(f, [f1, f2])
    assert f == q[0]*f1 + q[1]*f2 + r

def _test_s_polynomial():
    R = PolynomialRing(QQ, 'x, y', order='deglex')
    x,y = R.gens()
    f1 = x**3 - 2*x*y
    f2 = x**2*y - 2*y**2 + x
    f3 = s_polynomial(f1, f2)
    assert f3 == -x**2, f"S{f1, f2} = -x^2 != {f3}"
    f4 = s_polynomial(f1, f3)
    assert f4 == -2*x*y, f"S{f1, f3} = 2xy != {f4}"
    f5 = s_polynomial(f2, f3)
    assert f5 == x - 2*y**2, f"S{f2, f3} = x - 2y^2 != {f5}"
    # all combinations should be zero now -> f1, f2, f3, f4, f5 is a Gröbner basis
    for f, g in combinations([f1, f2, f3, f4, f5], 2):
        S = s_polynomial(f, g)
        assert polynomial_division(S, [f1, f2, f3, f4, f5])[1] == 0

    assert not is_groebner_basis([f1, f2, f3, f4])
    assert is_groebner_basis([f1, f2, f3, f4, f5])
    assert not is_minimal_groebner_basis([f1, f2, f3, f4, f5])
    assert not is_reduced_groebner_basis([f1, f2, f3, f4, f5])
    I = R.ideal([f1, f2])
    assert is_reduced_groebner_basis(I.groebner_basis())

    assert Buchberger([f1, f2]) == [f1, f2, f3, f4, f5]

def _test_elimination_ideal():
    R = PolynomialRing(QQ, 'x, y, z', order='lex')
    x,y,z = R.gens()
    I = R.ideal([x*y - 1, x*z - 1])
    assert elimination_ideal(I, x) == I.elimination_ideal(x)  # y - z

    # eliminate two variables
    I = R.ideal([x**2 + y**2 + z**2 - 1, x**2 + y**2 - z**2 - 1])
    assert elimination_ideal(I, [x,y]) == I.elimination_ideal([x, y])  # z^2

    R = PolynomialRing(CC, 'x, y, z', order='lex')
    x,y,z = R.gens()
    I = R.ideal([x*y - 1, x*z - 1])
    assert elimination_ideal(I, x) == R.ideal(y-z)  # I.elimination_ideal(x) doesn't work for CC

def _test_implicitization():
    g = lambda t,u: [t + u, t**2 + 2*t*u, t**3 + 3*t**2*u]  # polynomial parametric representation
    I = implicitization(g, 2, tnames='t,u', xnames='x,y,z')
    assert len(I) == 3
    assert len(I[0].gens()) == 3
    assert len(I[0].groebner_basis()) == 7
    assert len(I[1].gens()) == 6
    assert len(I[-1].gens()) == 1
    assert str(I[-1].gens()[0]) == '4*x^3*z - 3*x^2*y^2 - 6*x*y*z + 4*y^3 + z^2', f'Got {I[-1].gens()[0]}'

    r = lambda u,v: [u**2/v, v**2/u, u]  # rational parametric representation
    I = implicitization(r, 2, 'u, v', 'x, y, z')
    assert len(I[0].gens()) == 4
    assert len(I[0].groebner_basis()) == 8
    assert len(I[1].gens()) == 5
    assert len(I[-1].gens()) == 1
    assert str(I[-1].gens()[0]) == 'x^2*y - z^3', f'Got {I[-1].gens()[0]}'

def _test_reduction():
    R = PolynomialRing(QQ, 'x, y, z', order='lex')
    x,y,z = R.gens()
    f = x**4*y**3 + x**3*y**4
    assert reduction(f) == R.ideal(f).radical().gens()[0]

def _test_gcd():
    # integers
    assert gcd(2*3*7, 2*2*2*7) == 2*7
    assert gcd(2*3*7, 3*19) == 3
    assert gcd(42, 0) == 42
    assert gcd(0, 0) == 0
    assert gcd(12) == 12
    try:
        gcd([])
        assert False
    except:
        pass
    assert gcd(2,4,6,8) == 2
    assert gcd([2,4,6,8]) == 2
    assert gcd(range(0, 1000000, 10)) == 10

    # polynomials
    if sage_loaded:
        x, y = PolynomialRing(QQ, 'x, y').gens()
        assert gcd(x**6 - 1, x**4 - 1) == x**2 - 1

        f = 9*x**2*y**2 + 9*x*y**3 + 18*x**2*y + 27*x*y**2 - 9*y**3 + 9*x**2 + 27*x*y - 36*y**2 + 9*x - 45*y - 18
        g = 3*x**2*y+3*x*y**2 + 3*x**2 + 12*x*y+3*y**2 + 9*x + 9*y + 6
        assert gcd(f, g) == x*y + x + y**2 + 3*y + 2
    else:
        x = Polynomial([0, 1])
        assert gcd(x**6 - 1, x**4 - 1) == x**2 - 1

def _test_is_coprime():
    assert is_coprime(42, 57) == False
    assert is_coprime(42, 57, 13) == True

def _test_is_prime():
    assert is_prime(2) == True
    assert is_prime(1) == False
    assert is_prime(42) == False
    assert is_prime(43) == True
    # assert is_carmichael(997633) == True
    assert is_prime(997633) == False
    # assert is_prime(1000000000000066600000000000001) == True  # out of bounds
    # assert is_prime(512 * 2**512 - 1) == True  # out of bounds

def _test_prime_factors():
    assert prime_factors(12) == [2, 2, 3] and prime_factors(1) == []

def _test_euler_phi():
    assert euler_phi(1) == 1
    assert euler_phi(2) == 1
    assert euler_phi(10) == 4
    assert euler_phi(42) == 12

def _test_lcm():
    # Integers
    assert lcm(2*3*3, 2*2*3) == 2*2*3*3
    assert lcm(2*3*7, 3*19) == 2*3*7*19
    assert lcm(42, 0) == 0
    assert lcm(12) == 12
    assert lcm(2,4,6,8) == 24
    assert lcm([2,4,6,8]) == 24
    assert lcm(range(1, 50, 13)) == 7560
    try:
        lcm(0, 0)
        lcm([])
        assert False
    except:
        pass

    # Polynomials
    x = Polynomial([0, 1])
    assert lcm(x**3, x**2) == x**3
    assert lcm(x**2 + 1, x*2 + 2) == x**3 + x**2 + x + 1

def _test_closest_prime_factors_to():
    assert np.array_equal(closest_prime_factors_to(42, 13), [2, 7])

def _test_int_sqrt():
    assert int_sqrt(42) == 6
    assert int_sqrt(1) == 1
    assert int_sqrt(0) == 0

def _test_dlog():
    assert dlog(18, 2, 67) == 13
    assert dlog(17, 2, 67) == 64

def _test_is_carmichael():
    res = list(carmichael_numbers(2000))
    assert res == [561, 1105, 1729], f"Found {res} instead of [561, 1105, 1729]"
    for r in res:
        assert is_carmichael(r)
        assert not is_carmichael(r+2)

def _test_binFrac():
    assert binFrac(0.5, prec=12) == ".100000000000"
    assert binFrac(0.5, prec=0) == "."
    assert binFrac(np.pi-3, prec=12) == ".001001000011"

def _test_binstr_from_float():
    assert binstr_from_float(0) == "0"
    assert binstr_from_float(10) == "1010"
    assert binstr_from_float(0.5) == ".1"
    assert binstr_from_float(0.5, r=12) == ".100000000000"
    assert binstr_from_float(np.pi, r=20) == "11.00100100001111110111"
    assert binstr_from_float(0.5, r=0) == "0"  # https://mathematica.stackexchange.com/questions/2116/why-round-to-even-integers
    assert binstr_from_float(0.50000001, r=0) == "1"
    assert binstr_from_float(-3) == "-11"
    assert binstr_from_float(-1.5, r=3) == "-1.100"
    assert binstr_from_float(-0.125) == "-.001"
    assert binstr_from_float(-0.125, complement=True) == "-.111"
    assert binstr_from_float(-0.875, complement=False) == "-.111"
    assert binstr_from_float(0, r=3, complement=True) == ".000"
    assert binstr_from_float(-1.0, r=3, complement=True) == "-1.000"

def _test_float_from_binstr():
    assert np.allclose(float_from_binstr('1010'), 10)
    assert np.allclose(float_from_binstr('0'), 0)
    assert np.allclose(float_from_binstr('.100000000000'), 0.5)
    assert np.allclose(float_from_binstr('11.00100100001111110111'), np.pi)
    assert np.allclose(float_from_binstr('-11'), -3)
    assert np.allclose(float_from_binstr('-1.100'), -1.5)
    assert np.allclose(float_from_binstr('-.001'), -0.125)
    assert np.allclose(float_from_binstr('-.111', complement=True), -0.125)
    assert np.allclose(float_from_binstr('-.111', complement=False), -0.875)

    # check consistency of binstr_from_float and float_from_binstr
    assert np.allclose(float_from_binstr(binstr_from_float(0.5, r=2)), 0.5)
    assert np.allclose(float_from_binstr(binstr_from_float(-np.pi, r=20)), -np.pi, atol=1e-6)
    assert np.allclose(float_from_binstr(binstr_from_float(-0.375, r=3)), -0.375)
    assert np.allclose(float_from_binstr(binstr_from_float(-0.375, r=3, complement=True), complement=True), -0.375)

def _test_binstr_from_int():
    assert binstr_from_int(42) == "101010"
    assert binstr_from_int(0) == "0"
    assert binstr_from_int(1) == "1"

def _test_int_from_binstr():
    assert int_from_binstr("101010") == 42
    assert int_from_binstr("0") == 0
    assert int_from_binstr("1") == 1

def _test_binstr_from_bincoll():
    assert binstr_from_bincoll([1, 0, 1, 0, 1, 0]) == "101010"
    assert binstr_from_bincoll([0]) == "0"
    assert binstr_from_bincoll([1]) == "1"

def _test_bincoll_from_binstr():
    assert bincoll_from_binstr("101010") == [1, 0, 1, 0, 1, 0]
    assert bincoll_from_binstr("0") == [0]
    assert bincoll_from_binstr("1") == [1]

def _test_int_from_bincoll():
    assert int_from_bincoll([1, 0, 1, 0, 1, 0]) == 42
    assert int_from_bincoll([0]) == 0
    assert int_from_bincoll([1]) == 1

def _test_bincoll_from_int():
    assert bincoll_from_int(42) == [1, 0, 1, 0, 1, 0]
    assert bincoll_from_int(0) == [0]
    assert bincoll_from_int(1) == [1]

def _test_softmax():
    a = np.random.rand(5)
    b = softmax(a)
    assert np.isclose(np.sum(b), 1)

def _test_powerset():
    assert list(powerset([1,2,3])) == [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    assert list(powerset([])) == [()]

def _test_Fibonacci():
    assert Fibonacci(10) == 55
    assert Fibonacci(0) == 0
    assert Fibonacci(1) == 1

def _test_calc_pi():
    assert np.allclose(float(calc_pi1()), np.pi)
    assert np.allclose(float(calc_pi2()), np.pi)
    assert np.allclose(float(calc_pi3()), np.pi)
    assert np.allclose(float(calc_pi4()), np.pi)
    assert np.allclose(BBP_formula(1, 16, 8, [4, 0, 0, -2, -1, -1]), np.pi)
    assert np.allclose(1/2*BBP_formula(1, 2, 1, [1]), np.log(2))

def _test_log_():
    assert np.isclose(log_(42), np.log(42))
    assert np.isclose(log_(1234567890), np.log(1234567890))
    assert np.isclose(log_(2), np.log(2))
    assert np.isclose(log_(.000001), np.log(.000001))
    assert np.isclose(log_(np.e), 1)
    assert np.isclose(log_(1), 0)
    assert np.isclose(log_(0), -np.inf)

    assert np.isclose(log_2(42), np.log(42))
    assert np.isclose(log_2(1234567890), np.log(1234567890))
    assert np.isclose(log_2(2), np.log(2))
    assert np.isclose(log_2(.000001), np.log(.000001))
    assert np.isclose(log_2(np.e), 1)
    assert np.isclose(log_2(1), 0)
    assert np.isclose(log_2(0), -np.inf)