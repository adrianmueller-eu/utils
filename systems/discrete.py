import warnings
import numpy as np
import matplotlib.pyplot as plt
try:
    from scipy.linalg import eigvals
    # from scipy.signal import convolve2d
except ImportError:
    from numpy.linalg import eigvals
from tqdm import tqdm as tq

from .ode import get_roots
from .misc import plt_diagonal
from ..utils import startfile, ConvergenceCondition
from ..plot import colorize_complex
from ..mathlib import powerset, prime_factors

def flow_discrete_1d(f, x0s, lim=None, c=None, n_iter=1000, linewidth=.2, figsize=(10, 4), title=None, show=True):
    if isinstance(x0s, tuple) and len(x0s) == 2:
        x0s = np.linspace(x0s[0], x0s[1], 100)
    elif isinstance(x0s, tuple) and len(x0s) == 3:
        x0s = np.linspace(*x0s)
    xs = np.zeros((n_iter+1, len(x0s)))
    xs[0] = x0s
    for i in range(n_iter):
        xs[i+1] = f(xs[i])

    plt.figure(figsize=figsize)
    for i in range(len(x0s)):
        plt.plot(xs[:, i], color=c, linewidth=linewidth)
    plt.xlabel('$n$')
    plt.ylabel('$x_n$')
    plt.title(title)
    plt.xlim(0, n_iter)
    if lim is not None:
        plt.ylim(*lim)
    elif len(x0s) > 1:
        plt.ylim(np.min(x0s), np.max(x0s))
    plt.grid()
    if show:
        plt.show()

    return xs

def flow_discrete_2d(f, x0s, xlim=None, ylim=None, n_iter=1000, s=2, c=None, figsize=(10, 4), title=None, show=True):
    x0s = np.asarray(x0s)
    xs = np.zeros((n_iter+1, len(x0s)))
    ys = np.zeros((n_iter+1, len(x0s)))
    y0s = x0s[:,1]
    x0s = x0s[:,0]
    xs[0] = x0s
    ys[0] = y0s
    for i in range(n_iter):
        xs[i+1], ys[i+1] = f(xs[i], ys[i])

    plt.figure(figsize=figsize)
    for i in range(len(x0s)):
        if c is None:
            plt.scatter(xs[:,i], ys[:,i], c=range(n_iter+1), cmap='plasma', s=s)
        else:
            plt.scatter(xs[:,i], ys[:,i], c=c, s=s)
    plt.xlabel('$x_n$')
    plt.ylabel('$y_n$')
    plt.title(title)
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    elif len(x0s) > 1:
        plt.xlim(np.min(x0s), np.max(x0s))
        plt.ylim(np.min(y0s), np.max(y0s))
    plt.grid()
    if show:
        plt.show()

    return xs, ys

def orbit_diagram_discrete(f, x0s, rs, n_iter=1000, figsize=(10, 10), ylim=None, title=None, markersize=.1, show=True):
    plt.figure(figsize=figsize)
    if isinstance(rs, tuple) and len(rs) == 2:
        rs = np.linspace(rs[0], rs[1], 1000)
    elif isinstance(rs, tuple) and len(rs) == 3:
        rs = np.linspace(*rs)
    if isinstance(x0s, tuple) and len(x0s) == 2:
        x0s = np.linspace(x0s[0], x0s[1], 500)
    elif isinstance(x0s, tuple) and len(x0s) == 3:
        x0s = np.linspace(*x0s)
    for r in rs:
        x = x0s
        fr = f(r)
        for _ in range(n_iter):
            x = fr(x)
        plt.plot([r]*len(x), x, 'k.', markersize=markersize)
    plt.xlabel('r')
    plt.ylabel('x')
    plt.xlim(min(rs), max(rs))
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title(title)
    plt.grid()
    if show:
        plt.show()

def find_fixed_points_discrete(f, x0s, filter_eps=1e-6, distance_eps=1e-2):
    fpf = lambda x: f(x) - x
    return get_roots(fpf, x0s, filter_eps=filter_eps, distance_eps=distance_eps)

def classify_fixed_point_discrete(f, fp, eps=1e-6):
    if isinstance(fp, (int, float)):
        fp = [fp]
    fp = np.asarray(fp)
    # Calculate the Jacobian matrix
    dims = len(fp)
    J = np.zeros((dims, dims))
    for i in range(dims):
        eta = eps*np.eye(dims)[:,i]
        J[:, i] = (f(fp + eta) - f(fp - eta))/(2*eps)
    # Calculate the eigenvalues
    evals = eigvals(J)
    # Check the stability
    stable = np.all(np.abs(evals) < 1)
    classes = []
    for e in evals:
        if np.abs(e) < 1e-6:
            classes.append('superstable')
        elif np.abs(e - 1) < 1e-6 or np.abs(e + 1) < 1e-6:
            classes.append('nonlinear')
        elif np.abs(e) < 1:
            classes.append('stable')
        else:
            classes.append('unstable')
    return {'multipliers': evals, 'cls': classes, 'is_stable': stable}, stable

def coweb(f, x0s, max_iter=1000, eps=1e-6, xlim=None, ylim=None, title=None, figsize=(8,5), linewidth=.5,
          fp_res=100, fp_filter_eps=1e-6, fp_stability_eps=1e-6, fp_distance_eps=1e-3, verbose=True, show=True):
    # plot the cobweb diagram
    plt.figure(figsize=figsize)
    for i, x in enumerate(x0s):
        conv = ConvergenceCondition(max_iter=max_iter, eps=eps, skip_converged=2)
        xs = [x]
        ys = [0]
        while not conv(xs[-1]):
            try:
                x, x1 = xs[-1], f(x)
                assert np.isfinite(x1)
            except:
                x1 = np.nan
                break
            xs.extend([x, x1])
            ys.extend([x1, x1])
        # print(conv)
        if not np.isfinite(x1):
            warnings.warn(f"x0 = {x0s[i]} diverged ({x1})", stacklevel=2)
        else:
            plt.plot(xs, ys, linewidth=linewidth)

    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)

    extent = plt.axis()

    # find fixed points -> use fsolve to find roots of f - x
    xs = np.linspace(extent[0], extent[1], fp_res)
    fps = find_fixed_points_discrete(f, xs, filter_eps=fp_filter_eps, distance_eps=fp_distance_eps)
    for fp in fps:
        info, stable = classify_fixed_point_discrete(f, fp, eps=fp_stability_eps)
        if verbose:
            print(f"Fixed point {fp} ({','.join(info['cls'])}) {info['multipliers']}")
        if stable:
            # filled circle
            plt.plot(fp, fp, 'ko', markersize=5)
        else:
            # empty circle (unstable fixed point)
            plt.plot(fp, fp, 'wo', markersize=5, markeredgecolor='b')

    # draw f
    xs = np.linspace(extent[0], extent[1], 10*fp_res)
    plt.plot(xs, f(xs), 'b-', linewidth=1)

    plt_diagonal()
    plt.title(title)
    plt.xlabel('$x_n$')
    plt.ylabel('$x_{n+1}$')
    plt.grid()
    if show:
        plt.show()

    return fps

def p_iterate(f, p=2):
    if p == 1:
        return f
    def _p_iterate(x):
        for _ in range(p):
            x = f(x)
        return x
    return _p_iterate

def bifurcation_diagram_discrete(f, x0s, rs, p=1, dim=0, x_label='r', y_label='x', title='Bifurcation diagram',
        fp_filter_eps=1e-5, fp_distance_eps=1e-2, fp_stability_eps=1e-5, figsize=(8, 5), show=True):
    plt.figure(figsize=figsize)

    if isinstance(rs, tuple) and len(rs) == 2:
        rs = np.linspace(rs[0], rs[1], 200)
    elif isinstance(rs, tuple) and len(rs) == 3:
        rs = np.linspace(*rs)

    # for each p-period
    if isinstance(p, int):
        ps = list(range(1,p+1))
    else:
        ps = list(p)
    ps_filtered_periods = set()
    for p in sorted(set(ps), reverse=True):
        additionals = set([int(np.prod(p)) for p in powerset(prime_factors(p))])
        if len(additionals - ps_filtered_periods) != 0:
            ps_filtered_periods.update(additionals)
        else:
            ps.remove(p)
            print(f"Skipping {p}-period, already covered by some in {ps}")
    for p in tq(ps, desc='Periods', disable=len(ps) == 1):
        # get roots for each r
        all_roots = []
        for r in tq(rs, desc=f'Find roots {p}-period' if len(ps) > 1 else 'Find roots'):
            fr = p_iterate(f(r), p)
            roots = find_fixed_points_discrete(fr, x0s, filter_eps=fp_filter_eps, distance_eps=fp_distance_eps)
            # print(r, roots)
            all_roots.append(roots)

        # stability analysis
        stabilities = []
        if fp_stability_eps is not None:
            for r, roots in zip(rs, all_roots):
                fr = p_iterate(f(r), p)
                stabilities.append([classify_fixed_point_discrete(fr, root, eps=fp_stability_eps)[1] for root in roots])

        # scatter stable roots as black dots and unstable roots as red dots
        for r, roots, stables in zip(rs, all_roots, stabilities):
            for root, stable in zip(roots, stables):
                color = 'k' if stable else 'r'
                if not isinstance(root, (int, float)):
                    root = root[dim]
                plt.scatter(r, root, color=color, s=1)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(min(rs), max(rs))
    plt.grid()
    if show:
        plt.show()

    return rs, all_roots, stabilities

def Lyapunov_exponent_discrete(f, x0, f_derivative=None, max_iter=10000, finite_difference_eps=1e-6, approximation_eps=1e-6, skip_initial=300):
    conv = ConvergenceCondition(max_iter=max_iter, eps=approximation_eps, skip_converged=2, skip_initial=skip_initial)
    x = f(x0)
    sum_log = 0
    for i in range(max_iter):
        if f_derivative is not None:
            x_dev = f_derivative(x)
        else:
            # calculate the finite difference approximation of the derivative
            eps = finite_difference_eps
            x_dev = (f(x + eps) - f(x - eps))/(2*eps)
        term = np.log(np.abs(x_dev))
        if conv(term):
            break
        if i >= skip_initial and np.isfinite(term):
            sum_log += term
        x = f(x)
    return sum_log/(conv.iter-skip_initial)

def fractal(f, max_iters=100, res=1000, xlim=(-2,2), ylim=(-2,2), eps=None, min_points=1, kind='it', cmap='hot', save_fig=None, title="", colorbar=False, ticks=None, grid=False, show=True, ppi=60, **plt_kwargs):
    """ Plot a discrete differential equation on the complex plane. The equation
    `z = f(it, z, z0)` is iterated until convergence or `max_iters` iterations.

    Parameters
        f (function):    The function to iterate. Must take the arguments `it`, `z`, and `z0`.
        max_iters (int): The maximum number of iterations.
        res (int):       The number of pixels per unit length.
        xlim (tuple):    The limits along the real axis.
        ylim (tuple):    The limits along the imaginary axis.
        eps (float):     The maximum error allowed between iterations.
        min_points (int): Minimum amount of points that must not have diverged to continue.
        kind (str):      What to show. Can be 'it' (for the number of iterations needed for convergence), 'z' (for the actual complex numbers), or 'both'. If `save_fig` is set, `plt.show()` is not called.
        cmap (str):      The colormap to use when plotting the number of iterations.
        save_fig (str):  The name of the file to save the plot. If `None`, the plot is not saved.
        title (str):     The title of the plot.
        colorbar (bool): For the 'it' plot, whether to show the colorbar indicating the number of iterations until divergence.
        ticks (tuple):   Number of xticks and yticks to show. Set `None` to use the default and `0` to deactivate the axis.
        grid (bool):     Whether to show grid lines.
        show (bool):     If `show == True`, call `plt.show()`. If `save_fig` is set, alternatively specify `'open'` to start the system's default image viewer. If `save_fig` is not set and `show == 'gen'`, return the data as generator.
        ppi (int):       The resolution of the plot in pixels per inch.
        **plt_kwargs:    Keyword arguments passed to `plt.imshow`.

    Returns
        ndarray: The final values of `z`.
        ndarray: The number of iterations needed for convergence.
    """

    def _fractal(f, z0, max_iters, eps=1e-6):
        z = np.asarray(z0)
        error = np.inf
        iters = np.zeros_like(z, dtype='i4')
        for it in tq(range(1,max_iters+1)):
            if show == 'gen':
                yield z, iters
            z_ = f(it, z, z0)
            z_fin = np.isfinite(z_)
            iters[z_fin] = it
            z_[np.isnan(z_)] = np.inf
            if eps is not None:
                if np.sum(z_fin) == 0:
                    print(f"Chain diverged everywhere before {it} iterations")
                    break
                error_ = np.max(np.abs(z[z_fin] - z_[z_fin]))
                # print(it, "Error:",error_,np.abs(error - error_))
                if it > 1 and np.abs(error - error_) < eps:
                    print(f"Chain converged after {it} iterations (max error: {error_})")
                    break
                if it == max_iters:
                    print(f"Chain did not converge in {it} iterations (max error: {error_})")
                error = error_
            elif it % 100 == 0 and np.sum(z_fin) < min_points:
                print(f"Chain diverged everywhere before {it} iterations")
                break
            z = z_
        yield z, iters

    def get_res(xlim, ylim, res):
        """ Get the resolution of the plot. `res` is the number of pixels per unit length """
        x_d = xlim[1] - xlim[0]
        y_d = ylim[1] - ylim[0]
        if x_d < 1e-10:
            raise ValueError("xlim is too small or negative")
        if y_d < 1e-10:
            raise ValueError("ylim is too small or negative")
        x_res = int(res * np.sqrt(x_d / y_d))
        y_res = int(res * np.sqrt(y_d / x_d))
        print(f"Resolution: {x_res} x {y_res}")
        return x_res, y_res

    def get_meshgrid(xlim, ylim, x_res, y_res):
        """ Get the grid of points to iterate over """
        x = np.linspace(*xlim, x_res)
        y = np.linspace(*ylim, y_res)
        x, y = np.meshgrid(x, y)
        return x + 1j*y

    def get_figsize(x_res, y_res):
        """ Get the size of the figure """
        x_size = max(int(np.round(x_res / ppi)),1)
        y_size = max(int(np.round(y_res / ppi)),1)
        print(f"Figure size: {x_size} x {y_size}")
        return x_size, y_size

    x_res, y_res = get_res(xlim, ylim, res)
    mesh = get_meshgrid(xlim, ylim, x_res, y_res)
    if show == 'gen':
        return _fractal(f, mesh, max_iters, eps)
    else:
        z, iters = next(_fractal(f, mesh, max_iters, eps))

    # Plot the fractal
    def _plot(data, name):
        fig = plt.figure(figsize=get_figsize(x_res, y_res))
        img = plt.imshow(data, cmap=cmap, **plt_kwargs)
        plt.title(title)
        if colorbar and name == 'it':
            plt.colorbar(img, fraction=0.0312, pad=0.02)
        if ticks is None:
            plt.axis('off')
        else:
            if ticks[0] is not None:
                plt.xticks(np.linspace(0, x_res, ticks[0]), np.linspace(*xlim, ticks[0]))
            if ticks[1] is not None:
                plt.yticks(np.linspace(0, y_res, ticks[1]), np.linspace(*ylim, ticks[1]))
        if grid:
            plt.grid()
        if show == True:
            plt.show()
        if save_fig is not None:
            filename = save_fig + f'_{name}.png'
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            print(f"Saved to {filename}")
            if show == 'open':
                startfile(filename)

    if kind == 'it' or kind == 'both':
        _plot(iters, 'it')
    if kind == 'z' or kind == 'both':
        img = colorize_complex(z)
        _plot(img, 'z')

    return z, iters

def GoL(rows, cols, T, rule, init='zeros', progress=True):
    if isinstance(init, str):
        if init == 'zeros':
            grid = np.zeros((rows, cols), dtype='i1')
        elif init == 'ones':
            grid = np.ones((rows, cols), dtype='i1')
        elif init == 'random':
            grid = np.random.randint(2, size=(rows, cols), dtype='i1')
        else:
            raise ValueError(f'Invalid initializer: {init}')
    elif isinstance(init, np.ndarray):
        grid = init
    elif callable(init):
        grid = init(rows, cols)
    else:
        raise ValueError(f'Invalid initializer: {init}')
    assert grid.shape == (rows, cols), f'Invalid grid shape: {grid.shape} ≠ {(rows, cols)}'

    yield grid  # initial state
    for t in tq(range(T), disable=not progress):
        grid = rule(t, grid)
        yield grid

class Subgrid:

    def __init__(self, grid):
        self.grid = grid
        self._S, self._N = None, None

    @property
    def S(self):
        if self._S is None:
            self._S = np.roll(self.grid, 1, axis=0)
        return self._S

    @property
    def N(self):
        if self._N is None:
            self._N = np.roll(self.grid, -1, axis=0)
        return self._N

    @property
    def E(self):
        return np.roll(self.grid, 1, axis=1)

    @property
    def W(self):
        return np.roll(self.grid, -1, axis=1)

    @property
    def NE(self):
        return np.roll(self.N, 1, axis=1)

    @property
    def NW(self):
        return np.roll(self.N, -1, axis=1)

    @property
    def SE(self):
        return np.roll(self.S, 1, axis=1)

    @property
    def SW(self):
        return np.roll(self.S, -1, axis=1)

    @property
    def C(self):
        return self.grid

    def __call__(self, kernel):
        # return convolve2d(self.grid, kernel, mode='same', boundary='wrap')
        res = np.zeros_like(self.grid)
        if kernel[0,0]:
            res += self.NW
        if kernel[0,1]:
            res += self.N
        if kernel[0,2]:
            res += self.NE
        if kernel[1,0]:
            res += self.W
        if kernel[1,1]:
            res += self.grid
        if kernel[1,2]:
            res += self.E
        if kernel[2,0]:
            res += self.SW
        if kernel[2,1]:
            res += self.S
        if kernel[2,2]:
            res += self.SE
        return res