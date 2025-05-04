import sys
import numpy as np
from math import ceil, log10
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit, polyval, polyroots, polyadd, polysub, polyder, polyint, polymul
from abc import ABC, abstractmethod

# convenience functions
def pm(x, y=None, deg=1, plot=True, xlog=False, ylog=False):
    if y is None:
        if 1950 < x[0] < 2050 and all(1950 < x_i < 2050 for x_i in x[::2]):
            y = x[1::2]
            x = x[::2]
        else:
            y = x
            x = np.arange(len(y))
    if deg >= 0:
        poly = Polynomial.fit(x, y, deg)
    else:
        poly = InversePolynomial.fit(x, y, deg)
    if plot:
        x_ = np.linspace(min(x), max(x), 200)
        ax = poly.plot(x_)
        ax.scatter(x,y, marker=".")
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
        plt.show()
    return poly
    #return lambda x0: polyval(np.asarray(x0), coeffs)

def lm(x, y=None, plot=True):
    return pm(x, y, 1, plot)

# expm
def expm(x, y=None, plot=True, scaling_factor=True):
    if y is None:
        y = x
        x = np.arange(1, len(y)+1)
    exp = Exponential.fit(x, y, scaling_factor=scaling_factor)
    if plot:
        x_ = np.linspace(min(x), max(x), 200)
        ax = exp.plot(x_)
        ax.scatter(x,y)
    return exp

# logm
# sinm
# arm

# helper methods
def _to_str_coeff_1(c, precision=3, tol=0):
    if abs(c.imag) > tol:
        if abs(c.real) < tol:
            return f"{c.imag:.{precision}g}j"
        return f"({c.real:.{precision}g}{c.imag:+.{precision}g}j)"
    return f"{c.real:.{precision}g}"

def _generate_poly_label(coeffs, precision, tol=0):
    def _to_str(i):
        c = coeffs[i]
        if abs(c.imag) < tol:
            c = c.real
            if abs(c - int(np.round(c))) < tol:
                c = int(np.round(c))
        if i == 0:
            if type(c) == int:
                if c != 0:
                    return f"%d" % c
            elif abs(c) >= tol:
                return _to_str_coeff_1(c, precision, tol)
        elif i == 1:
            if type(c) == int:
                if c == 1:
                    return "x"
                elif c != 0:
                    return f"{c}x"
            elif abs(c) >= tol:
                return _to_str_coeff_1(c, precision, tol) + "*x"
        elif type(c) == int:
            if c == 1:
                return f"x**{i}"
            elif c != 0:
                return f"{c}*x**{i}"
        elif abs(c) >= tol:
            return _to_str_coeff_1(c, precision, tol) + f"x**{i}"
        else:
            print(f"Warning: coefficient {c} is too small.")
        return ""

    q = len(coeffs) - 1
    terms = [_to_str(i) for i in range(q, -1, -1)]
    res = " + ".join([t for t in terms if t != ""])
    res = res.replace(" + -", " - ")
    if res == "":  # zero polynomial
        res = "0"
    return res

# Functions
class Function(ABC):
    """
    Abstract class for functions. It provides an MSE error function and a plotting method.
    Each descendant has to implement the following methods:
    - fit(x, y)
    - __call__(x)
    - __str__()
    """

    @staticmethod
    @abstractmethod
    def fit(x, y):
        pass

    def mse(self, x, y):
        return np.mean(np.abs(self(x) - y)**2)

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def __str__(self, precision=3):
        pass

    def __repr__(self):
        return self.__str__()

    def label(self, precision=3, show_error=True):
        if show_error and hasattr(self, "error"):
            return self.__str__(precision) + f" (MSE: {self.error:.{precision}g})"
        return self.__str__(precision)

    def plot(self, x, ax=None, label='auto', precision=3):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        if label == 'auto':
            label = self.label(precision)
        ax.plot(x, self(x), label=label)
        if label is not None:
            ax.legend()
        return ax

class Polynomial(Function):
    """ y = a_n*x^n + ... + a_1*x + a_0 """

    PRINT_FACTORIZED = False

    def __init__(self, coeffs, tolerance=1e-7):
        assert len(coeffs) > 0, "Polynomial must have at least one coefficient."
        self.TOLERANCE = tolerance
        self.coeffs = tuple(coeffs)
        self.strip_coeffs()
        self._roots = None

    def copy(self):
        p = Polynomial(self.coeffs, self.TOLERANCE)
        p._roots = self._roots
        return p

    @property
    def degree(self):
        if self == 0:
            return np.inf
        return len(self.coeffs)-1

    @staticmethod
    def fit(x, y, deg=1):
        x = np.array(list(x))
        y = np.array(list(y))
        coeffs = polyfit(x, y, deg)
        p = Polynomial(coeffs)
        p.error = p.mse(x,y)
        return p

    @staticmethod
    def from_roots(roots):
        roots = list(roots)
        p = np.prod([Polynomial([-r, 1]) for r in roots])
        p._roots = roots
        return p

    def __call__(self, x):
        # return np.polyval(self.coeffs[::-1], x)
        # return np.vander(x, N=len(self.coeffs), increasing=True) @ self.coeffs
        assert not isinstance(x, Polynomial), "Did you forget a *?"
        # see np.polyval
        x = np.asarray(x)
        y = 0 if x.ndim == 0 else np.zeros_like(x)
        # Horner's method (...(c_n*x + c_{n-1})*x + ...)*x + c_0
        for pv in self.coeffs[::-1]:
            y *= x  # inplace manipulation
            y += pv
        return y

    def strip_coeffs(self):
        coeffs = list(self.coeffs)
        while abs(coeffs[-1]) <= self.TOLERANCE:
            coeffs.pop()
            if len(coeffs) == 0:
                coeffs = [0]
                break
        self.coeffs = tuple(coeffs)

    def normalize(self):
        self.coeffs = tuple(np.array(self.coeffs) / self.coeffs[-1])

    def is_monic(self):
        return abs(self.coeffs[-1] - 1) <= self.TOLERANCE

    def __str__(self, precision=3):
        if self.PRINT_FACTORIZED:
            return self.print_factorized(precision=precision)
        return _generate_poly_label(self.coeffs, precision, self.TOLERANCE)

    def plot(self, x=None, ax=None, label='auto'):
        if x is None:
            min_r, max_r = min(self.roots), max(self.roots)
            root_range = max_r - min_r
            x = np.linspace(-root_range*0.05, root_range*1.05, 1000) + min_r
        if ax is None:
            import matplotlib.pyplot as plt
            plt.figure()
            ax = plt.gca()
        ax.axhline(0, color='grey', linewidth=.5)
        return super().plot(x, ax, label=label)

    def __add__(self, other):
        if isinstance(other, Polynomial):
            return Polynomial(polyadd(self.coeffs, other.coeffs), self.TOLERANCE)
        elif isinstance(other, (int, float, complex)):
            new_coeffs = list(self.coeffs)
            new_coeffs[0] += other  # constant term
            return Polynomial(new_coeffs, self.TOLERANCE)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Polynomial):
            return Polynomial(polysub(self.coeffs, other.coeffs), self.TOLERANCE)
        elif isinstance(other, (int, float, complex)):
            new_coeffs = list(self.coeffs)
            new_coeffs[0] -= other  # constant term
            return Polynomial(new_coeffs, self.TOLERANCE)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Polynomial):
            return Polynomial(Polynomial._polymul(self.coeffs, other.coeffs), self.TOLERANCE)
        elif isinstance(other, (int, float, complex)):
            new_coeffs = [c*other for c in self.coeffs]
            return Polynomial(new_coeffs, self.TOLERANCE)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    @staticmethod
    def _polymul(coeffs1, coeffs2):
        """Multiply two polynomials given their coefficients."""
        return np.convolve(coeffs1, coeffs2)

    def __truediv__(self, other):
        if isinstance(other, Polynomial):
            # return Polynomial(polydiv(self.coeffs, other.coeffs))
            if other == 0:
                raise ZeroDivisionError("Division by zero.")
            return polynomial_division(self, other, verbose=False)
        elif isinstance(other, (int, float, complex)):
            new_coeffs = [c/other for c in self.coeffs]
            return Polynomial(new_coeffs, self.TOLERANCE)
        return NotImplemented

    def __floordiv__(self, other):
        res = self.__truediv__(other)
        if type(res) == tuple:
            return res[0]
        return res

    def __mod__(self, other):
        if isinstance(other, Polynomial):
            return self.__truediv__(other)[1]
        return NotImplemented

    def derivative(self, m=1):
        return Polynomial(polyder(self.coeffs, m), self.TOLERANCE)

    def integrate(self,m=1):
        return Polynomial(polyint(self.coeffs, m), self.TOLERANCE)

    def __eq__(self, other):
        if isinstance(other, Polynomial):
            return all(abs(c1 - c2) <= self.TOLERANCE for c1, c2 in zip(self.coeffs, other.coeffs))
        if isinstance(other, (int, float, complex)):
            return abs(self.coeffs[-1] - other) <= self.TOLERANCE and np.all([np.abs(self.coeffs[:-1]) <= self.TOLERANCE])
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __neg__(self):
        return Polynomial([-c for c in self.coeffs], self.TOLERANCE)

    def __pos__(self):
        return self

    def __pow__(self, n):
        if n == 0:
            return Polynomial([1], self.TOLERANCE)
        coeffs = list(self.coeffs)
        while n % 2 == 0:
            n //= 2
            coeffs = Polynomial._polymul(coeffs, coeffs)
        coeffs_ = coeffs
        while n > 1:
            coeffs = Polynomial._polymul(coeffs, coeffs_)
            n -= 1
        return Polynomial(coeffs, self.TOLERANCE)

    @property
    def roots(self):
        if self._roots is None:
            if self == 0:
                raise ValueError("The zero polynomial has roots everywhere!")

            # reduce the polynomial if by the minimum multiplicity of the roots
            p = self
            dp = p.derivative()
            m = 1
            g = p.gcd(dp)
            while g != 0 and g.degree > 1:
                m += 1
                second_last = p
                p = dp
                dp = p.derivative()
                g = p.gcd(dp)
            if m > 1:
                roots = []
                # print(m, p.roots)
                if p != 0:
                    p_ = self
                    for r in p.roots:  # work only if polyroots finds good roots for p
                        if abs(second_last(r)) <= p.TOLERANCE and all(abs(r - r_i) > p.TOLERANCE for r_i in roots):
                            r = np.round(r, -ceil(log10(p.TOLERANCE)))
                            # print(f"Root {r} has multiplicity {m}.")
                            if m == self.degree - 1:
                                m += 1
                            roots += [r]*m
                            p__, rem = p_ / Polynomial.from_roots([r]*m)
                            # assert rem == 0, (p_, p__, rem)
                            p_ = p__
                p = p_
                if p != self and p != 0:
                    roots += p.roots
                if len(roots) == self.degree and all(abs(self(r)) <= self.TOLERANCE for r in roots):  # check
                    # print(f"Found {len(roots)} roots: {roots}")
                    self._roots = roots
                    return roots
                # print(f"Found {len(roots)} ≠ {self.degree} roots: {roots}")

            potential_roots = polyroots(self.coeffs)
            roots = []
            for r_orig in potential_roots:
                # improve up to numerical precision using newton's method
                best_r = r = r_orig
                r_val = self(r)
                # use newton's method to find a value even closer to the actual root
                i, max_iter = 0, 100
                d = self.derivative()
                while abs(r_val) > self.TOLERANCE and i < max_iter: #+ sys.float_info.epsilon:
                    dr = d(r)
                    if abs(dr) < sys.float_info.epsilon:
                        break
                    r -= r_val / dr
                    r_val_new = self(r)
                    if abs(r_val_new) < abs(r_val):
                        best_r = r
                    r_val = r_val_new
                    i += 1
                # print(f"Root p({r_orig}) = {abs(self(r_orig))} improved to p({best_r}) = {abs(self(best_r))} in {i} iterations.")
                roots.append(best_r)
            self._roots = roots
            # self._roots = potential_roots
        return self._roots

    def variety(self, precision=None):
        if precision is not None:
            return set(np.round(self.roots, precision))
        return set(self.roots)

    def factors(self):
        roots = sorted(self.roots, key=lambda r: abs(r))
        return [Polynomial([-r, 1]) for r in roots]

    def print_factorized(self, precision=7):
        variety = self.variety(precision)
        variety = sorted(variety, key=lambda r: abs(r))
        roots = self.roots
        multiplicity = [roots.count(r) for r in variety]
        factors = []
        for r, m in zip(variety, multiplicity):
            factor = ""
            if np.isclose(r, 0):
                factor = "x"
            else:
                r = _to_str_coeff_1(r, precision, self.TOLERANCE)
                factor = f"(x-{r})"
                factor = factor.replace("--", "+")
            if m > 1:
                factor += f"**{m}"
            factors.append(factor)
        return "*".join(factors)

    def lt(self):
        return Polynomial([0]*self.degree + [self.coeffs[-1]])

    def is_monomial(self):
        return np.isclose(np.sum(np.abs(self.coeffs)), np.abs(self.coeffs[-1]))

    def divides(self, other):
        if self.degree > other.degree:
            return False
        if self.is_monomial() and other.is_monomial():
            return True
        return polynomial_division(other, self, verbose=False)[1] == 0

    def divisible(self, other):
        return other.divides(self)

    def gcd(self, other):
        if self == 0:
            return other
        if other == 0:
            return self
        if self.degree < other.degree:
            return other.gcd(self)
        old_r = Polynomial([0], self.TOLERANCE)
        while other != 0:
            r = self % other
            if r == old_r:
                return other
            old_r = r
            self, other = other, r
        return self

    def __bool__(self):
        return self != 0

def polynomial_division(f: Polynomial, g: Polynomial, verbose=True):
    """ Polynomial division. Returns the two polynomials q and r such that $f = qg + r$ and either r = 0 or deg(r) < deg(g). If r = 0, we say "g divides f". """
    assert g != 0, "Division by zero."

    # special case for monomial division
    if f.is_monomial() and g.is_monomial():
        coeffs = [0]*(f.degree - g.degree) + [f.coeffs[-1]/g.coeffs[-1]]
        return Polynomial(coeffs, f.TOLERANCE), Polynomial([0], f.TOLERANCE)

    q = Polynomial([0], f.TOLERANCE)
    r = f
    lt_g = g.lt()
    if verbose:
        i = 0
        print('q_0:', q)
        print('r_0:', r)
    while r != 0 and lt_g.divides(r.lt()):
        lt_rg, _ = r.lt() / lt_g
        q = q + lt_rg
        r = r - lt_rg * g
        if verbose:
            i += 1
            # print(f'lt_rg_{i}:', lt_rg)
            print(f'q_{i}:', q)
            print(f'r_{i}:', r)
    return q, r

x = Polynomial([0, 1])

class InversePolynomial(Function):
    """ y = 1/(a_n*x^n + ... + a_1*x + a_0) """

    def __init__(self, coeffs):
        self.coeffs = coeffs

    @property
    def degree(self):
        return 1-len(self.coeffs)

    @staticmethod
    def fit(x, y, deg=-1):
        x = np.array(list(x))
        y = np.array(list(y))
        coeffs = polyfit(x, 1/y, -deg)
        p = InversePolynomial(coeffs)
        p.error = p.mse(x,y)
        return p

    def __call__(self, x):
        return 1/polyval(np.asarray(x), self.coeffs)

    def __str__(self, precision=3):
        "1/(" + _generate_poly_label(self.coeffs, precision) + ")"

class Exponential(Function):
    """y = a*exp(b*(x-c)) + d"""

    def __init__(self, coeffs):
        if len(coeffs) != 4:
            raise ValueError("Exponential function needs exactly 4 coefficients: y = a*exp(b*(x-c)) + d")
        self.coeffs = coeffs

    @staticmethod
    def fit(x, y, scaling_factor=True):
        from scipy.optimize import curve_fit

        x = np.array(list(x))
        y = np.array(list(y))
        if scaling_factor:
            def func(x, a, b, c, d):
                return a*np.exp(b*(x-c)) + d
            coeffs, _ = curve_fit(func, x, y)
        else:
            def func(x, b, c, d):
                return np.exp(b*(x-c)) + d
            coeffs, _ = curve_fit(func, x, y)
            coeffs = [1]+list(coeffs)
        f = Exponential(coeffs)
        f.error = f.mse(x,y)
        return f

    def __call__(self, x):
        return self.coeffs[0]*np.exp(self.coeffs[1]*(x-self.coeffs[2])) + self.coeffs[3]

    def __str__(self, precision=12):
        b = f"{self.coeffs[1]:.{precision}g}"
        if b != '1':
            exponent = b + '*(' + f"x{-self.coeffs[2]:+.{precision}g}" + ')'
        else:
            exponent = f"x{-self.coeffs[2]:+.{precision}g}"
        a = f"{self.coeffs[0]:.{precision}g}"
        d = f"{self.coeffs[3]:+.{precision}g}"
        pre  = a + '*' if a != '1' else ''
        post = d if d[1:] != '0' else ''
        return pre+f"exp({exponent})"+post

# class Exponential(Function): # y = poly(exp(poly(x)))
# class Logarithm(Function): # y = poly(log_b(poly(x)))
# class Sine(Function): # y = poly(sin(poly(x)))
# class Autoregressive(Function): # x[t] = poly_i(x_i) for x_i in x[t-a:t]
