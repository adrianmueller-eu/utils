import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval, polyroots, polyadd, polysub, polyder, polyint, polymul
from abc import ABC, abstractmethod
from scipy.optimize import curve_fit

# convenience functions
def pm(x, y=None, deg=1, plot=True):
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
        #plt.show()
    return poly
    #return lambda x0: polyval(np.array(x0), coeffs)

def lm(x, y=None, plot=True):
    return pm(x, y, 1, plot)

# expm
def expm(x, y=None, plot=True):
    if y is None:
        y = x
        x = np.arange(1, len(y)+1)
    exp = Exponential.fit(x, y)
    if plot:
        x_ = np.linspace(min(x), max(x), 200)
        ax = exp.plot(x_)
        ax.scatter(x,y)
    return exp

# logm
# sinm
# arm

# helper methods
def _to_str_coeff_1(c, precision=3):
    if np.iscomplex(c):
        if np.isclose(c.imag, 0):
            return f"%.{precision}f" % c.real
        elif np.isclose(c.real, 0):
            return f"%.{precision}fj" % c.imag
        return f"(%.{precision}f%+.{precision}fj)" % (c.real, c.imag)
    return f"%.{precision}f" % c.real

def _generate_poly_label(coeffs, precision=3):
    def _to_str(i):
        c = coeffs[i]
        if np.isclose(c, int(c.real)):
            c = int(c.real)
        if i == 0:
            if type(c) == int:
                if c != 0:
                    return f"%d" % c
            elif not np.isclose(c, 0):
                return _to_str_coeff_1(c, precision)
        elif i == 1:
            if type(c) == int:
                if c == 1:
                    return "x"
                elif c != 0:
                    return f"{c}x"
            elif not np.isclose(c, 0):
                return _to_str_coeff_1(c, precision) + "x"
        elif type(c) == int:
            if c == 1:
                return f"x^{i}"
            elif c != 0:
                return f"{c}x^{i}"
        elif not np.isclose(c, 0):
            return _to_str_coeff_1(c, precision) + f"x^{i}"
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
            return self.__str__(precision) + f" (MSE: {self.error:.{precision}f})"
        return self.__str__(precision)

    def plot(self, x, ax=None, precision=3):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        ax.plot(x, self(x), label=self.label(precision))
        ax.legend()
        return ax

class Polynomial(Function):
    """ y = a_n*x^n + ... + a_1*x + a_0 """

    PRINT_FACTORIZED = False

    def __init__(self, coeffs):
        assert len(coeffs) > 0, "Polynomial must have at least one coefficient."
        self.coeffs = tuple(coeffs)
        self._roots = None

    @property
    def degree(self):
        if self.is_zero:
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

    def __call__(self, x):
        return polyval(np.array(x), self.coeffs)

    def __str__(self, precision=3):
        if Polynomial.PRINT_FACTORIZED:
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
                    r = _to_str_coeff_1(r, precision)
                    factor = f"(x-{r})"
                    factor = factor.replace("--", "+")
                if m > 1:
                    factor += f"^{m}"
                factors.append(factor)
            return "".join(factors)
        return _generate_poly_label(self.coeffs, precision)

    def __add__(self, other):
        if isinstance(other, Polynomial):
            return Polynomial(polyadd(self.coeffs, other.coeffs))
        elif isinstance(other, (int, float, complex)):
            new_coeffs = list(self.coeffs)
            new_coeffs[0] += other
            return Polynomial(new_coeffs)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Polynomial):
            return Polynomial(polysub(self.coeffs, other.coeffs))
        elif isinstance(other, (int, float, complex)):
            new_coeffs = list(self.coeffs)
            new_coeffs[0] -= other
            return Polynomial(new_coeffs)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Polynomial):
            return Polynomial(np.convolve(self.coeffs, other.coeffs))
        elif isinstance(other, (int, float, complex)):
            new_coeffs = [c*other for c in self.coeffs]
            return Polynomial(new_coeffs)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Polynomial):
            # return Polynomial(polydiv(self.coeffs, other.coeffs))
            if other.is_zero:
                raise ZeroDivisionError("Division by zero.")
            return polynomial_division(self, other, verbose=False)
        elif isinstance(other, (int, float, complex)):
            new_coeffs = [c/other for c in self.coeffs]
            return Polynomial(new_coeffs)
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
        return Polynomial(polyder(self.coeffs, m))

    def integrate(self,m=1):
        return Polynomial(polyint(self.coeffs,m))

    def gcd(self, other):
        while other:
            self, other = other, self % other
        return self

    def __eq__(self, other):
        if isinstance(other, Polynomial):
            return np.all(self.coeffs == other.coeffs)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __neg__(self):
        return Polynomial(-np.array(self.coeffs))

    def __pos__(self):
        return self

    def __pow__(self, n):
        if n == 0:
            return Polynomial([1])
        if n == 1:
            return self
        if n == 2:
            return self * self
        return self * (self**(n-1))

    @property
    def roots(self):
        if self._roots is None:
            self._roots = list(polyroots(self.coeffs))
        return self._roots

    def variety(self, precision=None):
        if precision is not None:
            return set(np.round(self.roots, precision))
        return set(self.roots)

    @property
    def factors(self):
        roots = self.roots
        roots = sorted(roots, key=lambda r: abs(r))
        factors = [Polynomial([-r, 1]) for r in roots]
        return factors

    @property
    def lt(self):
        return Polynomial([0]*self.degree + [self.coeffs[-1]])

    @property
    def is_monomial(self):
        return np.isclose(sum(self.coeffs), self.coeffs[-1])

    def divides(self, other):
        if self.degree > other.degree:
            return False
        if self.is_monomial and other.is_monomial:
            return True
        return polynomial_division(other, self, verbose=False)[1].is_zero

    def divisible(self, other):
        return other.divides(self)

    @property
    def is_zero(self):
        return np.allclose(self.coeffs, 0)

    def __bool__(self):
        return not self.is_zero

def polynomial_division(f: Polynomial, g: Polynomial, verbose=True):
    """ Polynomial division. Returns the two polynomials q and r such that $f = qg + r$ and either r = 0 or deg(r) < deg(g). If r = 0, we say "g divides f". """
    assert not g.is_zero, "Division by zero."

    # special case for monomial division
    if f.is_monomial and g.is_monomial:
        coeffs = [0]*(f.degree - g.degree) + [f.coeffs[-1]/g.coeffs[-1]]
        return Polynomial(coeffs), Polynomial([0])

    q = Polynomial([0])
    r = f
    lt_g = g.lt
    if verbose:
        i = 0
        print('q_0:', q)
        print('r_0:', r)
    while not r.is_zero and lt_g.divides(r.lt):
        lt_rg, _ = r.lt / lt_g
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
        return 1/polyval(np.array(x), self.coeffs)

    def __str__(self, precision=3):
        "1/(" + _generate_poly_label(self.coeffs, precision) + ")"

class Exponential(Function):
    """y = a*exp(b*(x-c)) + d"""

    def __init__(self, coeffs):
        if len(coeffs) != 4:
            raise ValueError("Exponential function needs exactly 4 coeffsicients")
        self.coeffs = coeffs

    @staticmethod
    def fit(x, y):
        x = np.array(list(x))
        y = np.array(list(y))
        def func(x, a, b, c, d):
            return a*np.exp(b*x+c) + d
        coeffs, _ = curve_fit(func, x, y)
        return Exponential(coeffs)

    def __call__(self, x):
        return self.coeffs[0]*np.exp(self.coeffs[1]*(x-self.coeffs[2])) + self.coeffs[3]

    def __str__(self, precision=3):
        exponent = f"{self.coeffs[1]:.{precision}f}*(x-{self.coeffs[2]:.{precision}f})"
        return f"{self.coeffs[0]:.{precision}f}*exp({exponent})+{self.coeffs[3]:.{precision}f}"

# class Exponential(Function): # y = poly(exp(poly(x)))
# class Logarithm(Function): # y = poly(log_b(poly(x)))
# class Sine(Function): # y = poly(sin(poly(x)))
# class Autoregressive(Function): # x[t] = poly_i(x_i) for x_i in x[t-a:t]
