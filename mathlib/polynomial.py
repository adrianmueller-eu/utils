import numpy as np
import itertools

from ..models import Polynomial
from .basic import choice

sage_loaded = False
try:
    from sage.all import *
    sage_loaded = True
except ImportError:
    pass


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
            for p, q in itertools.combinations(G, 2):
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
        for p, q in itertools.combinations(G, 2):
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
        for p, q in itertools.combinations(G, 2):
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
        for p, q in itertools.combinations(G, 2):
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
else:
    def random_polynomial(k=(1,10), c_range=(-1,1)):
        if hasattr(k, '__len__'):
            if len(k) == 2:
                k = np.random.randint(*k)
            else:
                k = choice(k)
        coeffs = np.random.random(k+1)*(c_range[1] - c_range[0]) + c_range[0]
        return Polynomial(coeffs)