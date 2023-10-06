import numpy as np
from math import factorial

from .utils import *

def entropy(p): # e.g. H(1*[1/2] + 4*[1/8])
    """Entropy!"""
    if callable(p):
        S = 0
        for px in p():
            if px > 0:
                S -= px*np.log2(px)
        return S
    if abs(np.sum(p) - 1) > 1e-10:
        raise ValueError("The overall probability must be 1!")
    return -sum((p*np.log2(p) for p in p if p > 0))  # 0*log(0) = 0

def smooth(y, smoothing=0.1):
    # https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    def savitzky_golay(y, window_size, order, deriv=0, rate=1):
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode='valid')

    # wilde guesses
    window = max(2,int(smoothing*len(y)))
    order = min(window-2,3)
    return savitzky_golay(y, window, order)

# converts 1-d data into a pdf, smoothing in [0,1]
def density(data, plot=False, label=None, smoothing=0.1, log=False, num_bins=None):
    import matplotlib.pyplot as plt

    if log:
        bins = logbins(data, scale=2, num=num_bins+1 if num_bins else bins_sqrt(data)+1)
        n, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = 10**(moving_avg(np.log10(bin_edges), 2))
    else:
        bins = num_bins or bins_sqrt(data)
        bins += 1
        n, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = moving_avg(bin_edges, 2)

    if smoothing:
        x, y = bin_centers, smooth(n, smoothing)
        # normalization
        dx = np.diff(bin_edges)
        y /= np.sum(y*dx)
    else:
        x,y = bin_centers, n

    if plot:
        if len(plt.get_fignums()) == 0:
            plt.figure(figsize=(10,5))
        plt.plot(x, y, label=label)
        top = max(plt.ylim()[1], 1.05*np.max(y))
        plt.ylim(bottom=0, top=top)
        plt.ylabel("Pdf")
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        if log:
            plt.xscale('log')
        if label:
            plt.legend()

    return x, y

# x,y should be an output of "density" above
def resample(x, y, size=int(1e6)):
    from scipy.interpolate import interp1d
    dx = np.diff(x)
    y_centers = moving_avg(y,2)
    y_cs = np.cumsum(y_centers*dx)
    invcdf = interp1d(y_cs, x, assume_sorted=True, bounds_error=False, fill_value=(0,1))
    u = np.random.uniform(0, 1, size)
    return invcdf(u)

def ste(ar):
    return np.std(ar, ddof=1) / np.sqrt(len(ar))


###############
### P #########
###############

# todo: make instance of scipy.stats.rv_continuous
# todo: save y in log-space
# todo: enable saving in log-x
# todo: implement for discrete x (pmf)
# todo: P.find_approximation(), which outputs a belief distribution p(model|data)
# todo: P.approx_normal(), P.approx_beta(), P.approx_lognormal(), ... (find best hyperparameters automatically)
# todo: write tests
class P:
    smoothing = 0.1
    resolution = int(1e4)
    rim = 1e-4
    resample_size = int(1e6)

    def __init__(self, x, y=None):
        import scipy

        if isinstance(x, scipy.stats._distn_infrastructure.rv_frozen):
            rv = x
            if rv.dist.__module__ == 'scipy.stats._continuous_distns':
                rim_h, rim_l = rv.isf([P.rim, 1-P.rim])
                x = np.linspace(rim_l, rim_h, P.resolution)
                y = rv.pdf(x) # normalized and smooth
            else:
                raise ValueError("Only continuous functions supported!")
        elif y is None:
            x, y = density(x, smoothing=P.smoothing) # smoothing and normalization included
        else:
            y = smooth(y, smoothing=P.smoothing)
            x, y = P._normalize(x,y)

        x, y = np.array(x), np.array(y)
        # pdf
        self.pdf = scipy.interpolate.interp1d(x, y, assume_sorted=True, bounds_error=False, fill_value=0)
        # cdf & inverse cdf
        dx = np.diff(x)
        y_centers = moving_avg(y,2)
        y_cs = np.cumsum(y_centers*dx)
        x_cs = moving_avg(x,2)
        self.cdf = scipy.interpolate.interp1d(x_cs, y_cs, assume_sorted=True, bounds_error=False, fill_value=(0,1))
        self.invcdf = scipy.interpolate.interp1d(y_cs, x_cs, assume_sorted=True, fill_value="extrapolate") # todo: bad approximation in the asymptotes
        self.quantile = self.invcdf # alias

    @property
    def x(self):
        return self.pdf.x

    @property
    def y(self):
        return self.pdf.y

    def __op__(self, other, op):
        x = np.concatenate([self.x,other.x])
        x.sort()
        y = op(x)
        return P(x,y) # normalizes in constructor

    def __add__(self, other):
        return self.__op__(other, lambda x: self(x)+other(x))

    def __sub__(self, other):
        return self.__op__(other, lambda x: self(x)-other(x))

    def __mul__(self, other):
        return self.__op__(other, lambda x: self(x)*other(x))

    def __truediv__(self, other):
        return self.__op__(other, lambda x: self(x)/other(x))

    def __matmul__(self, other):
        return self.__op__(other, lambda x: np.convolve(self(x), other(x), mode='same'))

    def __call__(self, x):
        return self.pdf(x)

    def mean(self):
        return np.average(self.x, weights=self.y)

    def median(self):
        return self.invcdf(0.5)

    def mode(self):
        return self.x[np.argmax(self.y)]

    def plot(self, show=False, *pltargs, **pltkwargs):
        import matplotlib.pyplot as plt

        plt.plot(self.x, self.y, *pltargs, **pltkwargs)
        if show:
            plt.show()

    def sample(self, size=1):
        u = np.random.uniform(0, 1, size)
        return self.invcdf(u)

    def resample(self, size=None):
        if not size:
            size = P.resample_size
        return self.sample(size)

    @property
    def nbytes(self):
        n_pdf = self.pdf.x.nbytes
        n_cdf = self.cdf.x.nbytes
        n_invcdf = self.invcdf.x.nbytes
        return n_pdf*2 + n_cdf*2 + n_invcdf*2 # each x and y

    @staticmethod # e.g. b = P.use(lambda p: binom.pmf(44, 274, p))
    def use(f, start=0, stop=1, size=None):
        if not size:
            size = P.resolution
        x = np.linspace(start, stop, size)
        y = [f(i) for i in x]
        return P(x,y) # normalizes in constructor

    @staticmethod
    def _normalize(x, y):
        dx = np.diff(x)
        y_centers = moving_avg(y,2)
        integral = np.sum(dx*y_centers)
        if integral < 1e-20 or integral > 1e20 or np.isnan(integral):
            raise ValueError(f"Not normalizable! Integral was %s" % integral)
        y = y/integral # copy
        return x, y



