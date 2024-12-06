from .basic import *
from .binary import *
from .sets import *
from .matrix import *
from .polynomial import *
from .number_theory import *
from .misc import *

#############
### Tests ###
#############

import warnings
from numpy.random import randint

def test_mathlib_all():
    tests = [
        # basic
        _test_series,
        _test_sequence,
        _test_rad,
        _test_deg,
        _test_softmax,
        _test_choice,
        # set
        _test_bipartitions,
        # binary
        _test_binFrac,
        _test_binstr_from_float,
        _test_float_from_binstr,
        _test_binstr_from_int,
        _test_int_from_binstr,
        _test_bincoll_from_binstr,
        _test_binstr_from_bincoll,
        _test_int_from_bincoll,
        _test_bincoll_from_int,
        # matrix
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
        _test_is_diag,
        _test_matexp,
        _test_matlog,
        _test_normalize,
        _test_immanant,
        # polynomial
        _test_roots,
        # number_theory
        _test_gcd,
        _test_is_coprime,
        _test_closest_prime_factors_to,
        _test_int_sqrt,
        _test_dlog,
        _test_is_carmichael,
        # misc
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

def _test_rad():
    assert rad(180) == np.pi
    assert rad(0) == 0

def _test_deg():
    assert deg(np.pi) == 180
    assert deg(0) == 0

def _test_bipartitions():
    actual = list(bipartitions(range(3), unique=True))
    expected = [([0], [1, 2]), ([1], [0, 2]), ([2], [0, 1])]
    for a in actual:
        assert a in expected or a[::-1] in expected  # order doesn't matter
    assert len(actual) == len(expected)

    actual = list(bipartitions(list('abc'), unique=False))
    expected = [(['a'], ['b', 'c']),
                (['b'], ['a', 'c']),
                (['c'], ['a', 'b']),
                (['a', 'b'], ['c']),
                (['a', 'c'], ['b']),
                (['b', 'c'], ['a'])]
    for a in actual:
        assert a in expected
    assert len(actual) == len(expected)

def _test_powerset():
    assert list(powerset([1,2,3])) == [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    assert list(powerset([])) == [()]

def _test_softmax():
    a = np.random.rand(5)
    b = softmax(a)
    assert np.isclose(np.sum(b), 1)

def _test_choice():
    a = [0,1,2,3]  # list
    assert choice(a) in a
    a = np.random.rand(5)  # np.ndarray
    assert choice(a) in a
    a = np.random.rand(5,4)  # vectors
    assert choice(a) in a
    a = [i for i,o in bipartitions(range(5))]  # different lengths
    assert choice(a) in a
    x = Polynomial([0, 1])
    a = [x**i for i in range(4)]  # objects
    assert choice(a) in a

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

def _test_is_symmetric():
    a = [
        [1, 2],
        [2, 1]
    ]
    assert is_symmetric(a)

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
    a = random_hermitian(randint(2,20))
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
    U = A @ inv(J)  # polar decomposition
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
    d = randint(2,20)
    U = random_unitary(d)
    p = np.random.rand(d)
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
    if not np.allclose(P, np.eye(n)):
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

def _test_is_diag():
    a = np.eye(randint(2,20))
    assert is_diag(a)

    a = random_square(randint(2,20))
    assert not is_diag(a)

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

def _test_normalize():
    a = random_vec(randint(2,20), complex=True)
    b = normalize(a)
    assert np.isclose(np.linalg.norm(b), 1)
    a = np.array(3 - 4j)
    assert np.isclose(normalize(a), a/5)

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