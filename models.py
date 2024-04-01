import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval
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
    #return lambda x0: polyval(np.array(x0), coeff)

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
def _generate_poly_label(coeff, precision=3):
    q = len(coeff) - 1
    res = ""
    for i,c in enumerate(reversed(coeff)):
       if q-i == 0:
          res += f"%.{precision}f" % c
       else:
          res +=f"%.{precision}fx^%d + " % (c, q-i)
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

    def __init__(self, coeff):
        self.coeff = coeff

    @property
    def degree(self):
        return len(self.coeff)-1

    @staticmethod
    def fit(x, y, deg=1):
        x = np.array(list(x))
        y = np.array(list(y))
        coeff = polyfit(x, y, deg)
        p = Polynomial(coeff)
        p.error = p.mse(x,y)
        return p

    def __call__(self, x):
        return polyval(np.array(x), self.coeff)

    def __str__(self, precision=3):
        return _generate_poly_label(self.coeff, precision)

class InversePolynomial(Function):
    """ y = 1/(a_n*x^n + ... + a_1*x + a_0) """

    def __init__(self, coeff):
        self.coeff = coeff

    @property
    def degree(self):
        return 1-len(self.coeff)

    @staticmethod
    def fit(x, y, deg=-1):
        x = np.array(list(x))
        y = np.array(list(y))
        coeff = polyfit(x, 1/y, -deg)
        p = InversePolynomial(coeff)
        p.error = p.mse(x,y)
        return p

    def __call__(self, x):
        return 1/polyval(np.array(x), self.coeff)

    def __str__(self, precision=3):
        "1/(" + _generate_poly_label(self.coeff, precision) + ")"

class Exponential(Function):
    """y = a*exp(b*(x-c)) + d"""

    def __init__(self, coeff):
        if len(coeff) != 4:
            raise ValueError("Exponential function needs exactly 4 coefficients")
        self.coeff = coeff

    @staticmethod
    def fit(x, y):
        x = np.array(list(x))
        y = np.array(list(y))
        def func(x, a, b, c, d):
            return a*np.exp(b*x+c) + d
        coeff, _ = curve_fit(func, x, y)
        return Exponential(coeff)

    def __call__(self, x):
        return self.coeff[0]*np.exp(self.coeff[1]*(x-self.coeff[2])) + self.coeff[3]

    def __str__(self, precision=3):
        exponent = f"{self.coeff[1]:.{precision}f}*(x-{self.coeff[2]:.{precision}f})"
        return f"{self.coeff[0]:.{precision}f}*exp({exponent})+{self.coeff[3]:.{precision}f}"

# class Exponential(Function): # y = poly(exp(poly(x)))
# class Logarithm(Function): # y = poly(log_b(poly(x)))
# class Sine(Function): # y = poly(sin(poly(x)))
# class Autoregressive(Function): # x[t] = poly_i(x_i) for x_i in x[t-a:t]
