import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval
from abc import ABC, abstractmethod
from scipy.optimize import curve_fit

# convenience functions
def pm(x, y=None, deg=1, plot=True):
    if y is None:
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
def _generate_poly_label(coeff):
    q = len(coeff) - 1
    res = ""
    for i,c in enumerate(reversed(coeff)):
       if q-i == 0:
          res += "%.3f" % c
       else:
          res += "%.3fx^%d + " % (c, q-i)
    return res

# Functions
class Function(ABC):
    """
    Abstract class for functions. It provides an MSE error function and a plot method.
    Each descendant has to implement the following methods:
    - fit(x, y)
    - __call__(x)
    - __str__()
    - _plot_label()
    """

    @staticmethod
    @abstractmethod
    def fit(x, y):
        pass

    def error(self, x, y):
        return np.mean(np.abs(self(x) - y)**2)

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def _plot_label(self):
        pass

    def plot(self, x, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        ax.plot(x, self(x), label=self._plot_label())
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
        p.fit_err = p.error(x,y)
        return p

    def __call__(self, x):
        return polyval(np.array(x), self.coeff)

    def __str__(self):
        return f"Polynomial of degree %d with coeff %s" % (self.degree, self.coeff)

    def _plot_label(self):
        return _generate_poly_label(self.coeff)

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
        p.fit_err = p.error(x,y)
        return p

    def __call__(self, x):
        return 1/polyval(np.array(x), self.coeff)

    def __str__(self):
        return f"Inverse polynomial of degree %d with coeff %s" % (self.degree, self.coeff)

    def _plot_label(self):
        return "1/(" + _generate_poly_label(self.coeff) + ")"

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

    def __str__(self):
        return f"Exponential function with coeff {self.coeff}"

    def _plot_label(self):
        return f"{self.coeff[0]:.3f}*exp({self.coeff[1]:.3f}*(x-{self.coeff[2]:.3f}))+{self.coeff[3]:.3f}"


# class Exponential(Function): # y = poly(exp(poly(x)))
# class Logarithm(Function): # y = poly(log_b(poly(x)))
# class Sine(Function): # y = poly(sin(poly(x)))
# class Autoregressive(Function): # x[t] = poly_i(x_i) for x_i in x[t-a:t]
