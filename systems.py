import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.optimize import fsolve
from itertools import product, combinations
from tqdm.auto import tqdm as tq

from .plot import colorize_complex
from .utils import ConvergenceCondition
from .mathlib import powerset, prime_factors

## This script contains functions to visualize and analyze dynamical systems
## So far this only covers 1D and 2D first-order ODEs, as well as iterative maps (fractals!)

############
### Flow ###
############

def simulate(f, x0, T, dt, method='RK45'):
    sol = solve_ivp(lambda t, state: f(*state), y0=x0, method=method, t_span=(0.0, T), dense_output=True)
    t = np.arange(0, T, dt)
    return *sol.sol(t), t

def ODE_flow_1d(f, x0s=None, xlim=(-2,2), T=10, n_timesteps=10000, show_arrows=True, grid=False, dim=None, figsize=(6,3), c=None, linewidth=1, ax=None, title="Flow", show=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x_min, x_max = xlim
    dt = T/n_timesteps
    if dim is None and show_arrows:
        t, x = np.meshgrid(np.arange(0, T, T/20), np.arange(x_min, x_max, (x_max - x_min)/20))
        V = f(x)
        U = np.ones(V.shape)

        # slope field
        q = ax.quiver(t, x, U, V, angles='xy')
        ax.quiverkey(q, X=0.8, Y=1.03, U=4, label='dx/dt', labelpos='E')
        ax.set_ylim(*ax.get_ylim())
        dim = 0
    else:
        if dim is None:
            dim = 0
        ax.set_ylim(x_min, x_max)

    # trajectories
    if x0s is not None:
        for x0 in x0s:
            # convert x0 to a list if it is a scalar
            if isinstance(x0, (int, float)):
                x0 = [x0]
            s = simulate(f, x0, T, dt)
            x, t = s[dim], s[-1]
            ax.plot(t, x, c=c, linewidth=linewidth)

    ax.set_xlim(-0.02*T, T)
    if grid:
        ax.grid()
    ax.set_title(title)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    if show:
        plt.show()

def ODE_flow_2d(f, x0s=None, xlim=(-2,2), ylim=(-2,2), T=10, n_timesteps=10000, ax=None, title="Flow"):
    dt = T/n_timesteps

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    if x0s is not None:
        for x0 in x0s:
            x, y, t = simulate(f, x0, T, dt)
            ax.plot(t,x,y)
            ax.scatter(0, *x0, marker="x")

    ax.set_title(title)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_xlim(0, T)
    ax.set_ylim(*xlim)
    ax.set_zlim(*ylim)
    plt.show()


######################
### Phase portrait ###
######################

def is_stable(f, fp, T=100, dt=1, eps=1e-3, verbose=False):
    fp = np.array(fp)
    dims = len(fp)
    noise = np.random.normal(0, 1, dims)  # normal = move to random direction
    noise *= eps/np.linalg.norm(noise)    # length eps -> uniform point on the eps-sphere
    x = simulate(f, fp+noise, T, dt)
    # get the last value of the simulation
    res = [x[i][-1] for i in range(dims)]
    error = np.linalg.norm(res - fp)
    stable = error <= eps  # stable if it remains inside eps
    if verbose:
        s = '<=' if stable else '>'
        if dims == 2:
            cls, info = classify_fixed_point(f, fp, eps, verbose=False)
            print(f"Fixed point ({cls}) {fp}: {stable} (stability eps: {error} {s} {eps})")
        else:
            print(f"Fixed point {fp}: {stable} (stability eps: {error} {s} {eps})")
    return stable

def classify_fixed_point_2D(l1, l2, J=None):
    tr = (l1 + l2).real
    det = (l1*l2).real
    dis = tr**2 - 4*det
    is_stable = det >= 0 and tr <= 0  # both eigenvalues have negative real part

    err = 1e-10
    if det < -err:  # leave space for the non-isolated fixed point
        cls = "Saddle"
    elif dis < -err:  # leave space for degenerate nodes
        if tr > err:
            cls = "Unstable spiral"
        elif tr < -err:
            cls = "Stable spiral"
        else:  # dis < 0 and tr = 0 -> imaginary eigenvalues
            cls = "Center"
    elif det > err and np.abs(tr) > err:  # leave space for the non-isolated fixed points
        if dis > err:
            if tr > err:
                cls = "Unstable node"
            elif tr < -err:
                cls = "Stable node"
        else:  # np.abs(dis) < err:  # l1 == l2
            if J is not None:
                if np.abs(J[0,1]) < err and np.abs(J[1,0]) < err:  # J = lambda*I
                    if tr > err:
                        cls = "Unstable star node"
                    elif tr < -err:
                        cls = "Stable star node"
                else:
                    if tr > err:
                        cls = "Unstable degenerate node"
                    elif tr < -err:
                        cls = "Stable degenerate node"
            else:
                if tr > err:
                    cls = "Unstable degenerate/star node"
                elif tr < -err:
                    cls = "Stable degenerate/star node"
    elif np.abs(tr) > err:  # and np.abs(det) < err => one eigenvalue is zero
        cls = "Non-isolated fixed points (line)"
    else:  # tr = det = 0 -> J = 0
        cls = "Non-isolated fixed points (plane)"
    return cls, {'tr': tr, 'det': det, 'dis': dis, 'stable': is_stable}

def classify_fixed_point(f, fp, eps, verbose=True):
    """
    Classify a fixed point of a nD ODE system using the local derivatives at the fixed point (linearization).
    - For 1D, it returns 'Stable', 'Unstable', 'Left half-stable' or 'Right half-stable'.
    - For 2D ODEs (using the Jacobian matrix), it returns the classification of the fixed point and whether it is stable. 
    By Hartman-Grobman theorem, this generally works well for hyperbolic fixed points (saddle, nodes, spirals),
    but not for centers or non-isolated fixed points (imaginary eigenvalues).
    - For nD ODEs, n>2, it returns the classification of the fixed point in all 2-combinations of the eigenvalues.

    Args:
    `f` (function):         The system of ODEs. Takes n arguments and returns n derivatives.
    `fp` (float or tuple):  The fixed point to classify
    `eps` (float):          The precision of the finite differences
    `verbose` (bool):       Whether to print the classification

    Returns:
    str|list[str]:        The classification(s) of the fixed point
    bool:                 Whether the fixed point is stable in all dimensions
    """
    if isinstance(fp, (int, float)) or len(fp) == 1:
        x = np.array(fp)
        # Estimate the derivative at the fixed point via finite differences
        ffp = f(x)
        df_l = (f(x - eps) - ffp)/eps
        df_r = (f(x + eps) - ffp)/eps
        stable = (df_r - df_l)/2 <= 0 # (f(x + eps) - f(x - eps))/(2*eps) < 0
        if df_l is None or df_r is None:
            cls = "Unknown"
        if df_l >= 0 and 0 >= df_r:
            cls = "Stable"
        elif df_l < 0 and 0 < df_r:
            cls = "Unstable"
        elif df_l < 0 and 0 > df_r:
            cls = "Right half-stable"
        elif df_l > 0 and 0 < df_r:
            cls = "Left half-stable"
        else:
            cls = f"WTF? df_r={df_r}, df_l={df_l}"
        if verbose:
            print(f"Fixed point {fp}: {cls} ({np.round(ffp,3)})")
        return cls, stable

    # Estimate Jacobian matrix at the fixed point via finite differences
    if verbose:
        ffp = np.array(f(*fp))
    fp = np.array(fp)
    dims = len(fp)
    J = np.zeros((dims,dims))
    for k in range(dims):
        # partial derivative of f with respect to x_k at the root
        dfabu = np.array(f(*(fp + eps*np.eye(dims)[:,k])))
        dfabl = np.array(f(*(fp - eps*np.eye(dims)[:,k])))
        J[:,k] = (dfabu - dfabl)/(2*eps)
    if np.any(np.isnan(J)):
        if verbose:
            print(f"Fixed point {fp}: Unknown ({np.round(ffp,3)})")
        return "Unknown", False
    # all 2-combinations of the eigenvalues (use itertools.combinations)
    if dims == 2:
        cls, info = classify_fixed_point_2D(*np.linalg.eigvals(J), J)
        if verbose:
            print(f"Fixed point {fp}: {cls} ({np.round(ffp,3)}), tr={info['tr']}, det={info['det']}, dis={info['dis']}")
        return cls, info['stable']

    # dims > 2
    classes = []
    stable = True
    if verbose:
        print(f"Fixed point {fp}: ({np.round(ffp,3)})")
    for dims, (l1, l2) in zip(combinations(range(len(fp)), 2), combinations(np.linalg.eigvals(J), 2)):
        cls, info = classify_fixed_point_2D(l1, l2)
        if verbose:
            print(f"  {dims}: {cls}, tr={info['tr']}, det={info['det']}, dis={info['dis']}")
        if not info['stable']:
            stable = False
        classes.append((dims, (cls, info)))

    return classes, stable

def find_fixed_points(f, dots, mins, ds, filter_eps, distance_eps):
    # sanity check
    assert callable(f), "f must be a function"
    assert len(dots) == len(mins) == len(ds), "The number of dimensions must match the number of dots, mins and ds"

    idcs = np.where(np.logical_and.reduce([np.abs(dot) < filter_eps for dot in dots]))

    # first, sort fps by closeness into lists
    fps_lists = []
    for idx in zip(*idcs):
        if len(dots) == 1:
            fp = [mins[0] + idx[0]*ds[0]]
        elif len(dots) == 2:
            fp = [mins[0] + idx[1]*ds[0], mins[1] + idx[0]*ds[1]]
        elif len(dots) == 3:
            fp = [mins[0] + idx[1]*ds[0], mins[1] + idx[0]*ds[1], mins[2] + idx[2]*ds[2]]
        else:
            raise ValueError("Only 1D, 2D and 3D supported")

        # check if it actually is a fixed point, i.e. if the other dimensions are close to zero, as well
        if np.max(np.abs(f(*fp))) > filter_eps:
            continue
        found = False
        for j, fps_list in enumerate(fps_lists):
            # if np.linalg.norm(np.array(fps_list[0]) - np.array(fp)) < eps:
            if np.mean(np.linalg.norm(np.array(fps_list) - np.array(fp), axis=1)) < distance_eps:
                fps_list.append(fp)
                found = True
                break
        if not found:
            fps_lists.append([fp])
    # then, find for each list the fp, such that dots are closest to 0
    fps = []
    for fps_list in fps_lists:
        # evaluate f at all fps
        dots = np.array([f(*fp) for fp in fps_list])
        if len(dots.shape) == 1:
            dots = dots[:,None]
        # find the closest point to 0
        errors = np.linalg.norm(dots, axis=1)
        idx = np.argmin(errors)
        fps.append(fps_list[idx])
    return fps

def ODE_phase_1d(f, xlim=(-2,2), dims=(None,), T=20, n_timesteps=4000, ax=None, n_arrows=10, x_label="x", title="Phase portrait", figsize=(8,4), grid=False, show=True, ylim=None,
                 fp_resolution=1000, fp_filter_eps=1e-3, fp_distance_eps=1e-1, stability_method='jacobian', fp_stability_eps=1e-2):
    """
    Phase portrait of a first-order 1D ODE

    Hints:
    - If there are multiple fixed points stacked on top of each other, try increasing `fp_distance_eps`

    Args:
    `f` (function):             The ODE. Must take one argument, `x`, and return the first derivative `x_dot`
    `xlim` (tuple):             The limits of the x-axis
    `dims` (tuple):             If the system has more than one dimension, select which dimension to plot by setting it to `None` and provide a default value for the other dimensions.
    `T` (float):                The time to simulate the trajectories and check the stability of the fixed points
    `n_timesteps` (int):        The number of timesteps to simulate the trajectories in `[0, T]`
    `ax` (matplotlib axis):     Optional axis to plot the phase portrait
    `n_arrows` (int):           The number of arrows in the x axis
    `x_label` (str):            The label of the x axis
    `title` (str):              The title of the plot
    `figsize` (tuple):          The size of the plot
    `grid` (bool):              Whether to show a grid
    `show` (bool):              Whether to show the plot
    `fp_resolution` (int):      The resolution of the grid in which to look for fixed points and plot the slope field
    `fp_filter_eps` (float):    The maximum `|f(x^*)|` for which `x^*` is considered a fixed point
    `fp_distance_eps` (float):  The maximum distance between fixed points for them to be considered the same.
    `stability_method` (str):   The method to check for stability: 'jacobian' or 'lyapunov'.
    `fp_stability_eps` (float): The precision when checking for stability. If method is `jacobian`, it controls the precision of the finite differences and for method `lyapunov` the distance of the test particle.
    """

    x_min, x_max = xlim
    dt = T/n_timesteps
    if n_arrows is None or n_arrows < 1:
        n_arrows = 0
        dx = (x_max - x_min)/(fp_resolution*10)
    else:
        dx = (x_max - x_min)/(fp_resolution*n_arrows)
    x = np.arange(x_min, x_max, dx)
    xdim = dims.index(None)
    if len(dims) > 1:
        def f_1d(x):
            args = list(dims)
            args[xdim] = x
            return f(*args)
    else:
        f_1d = f
    x_dot = f_1d(x)

    # plot the function
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, x_dot)

    # round and set
    fps = find_fixed_points(f_1d, [x_dot], [x_min], [dx], fp_filter_eps, fp_distance_eps)
    for fp in fps:
        _fp = list(dims)
        _fp[xdim] = fp
        if stability_method == 'jacobian':
            cls, stable = classify_fixed_point(f, _fp, fp_stability_eps, verbose=True)
        elif stability_method == 'lyapunov':
            stable = is_stable(f, _fp, T, dt, fp_stability_eps, verbose=True)
            cls = 'Stable' if stable else 'Unstable'
        else:
            raise ValueError(f"stability_method must be 'lyapunov' or 'jacobian', not {stability_method}")

        if isinstance(cls, str):
            if cls == 'Stable':
                fillstyle = 'full'
            elif cls == 'Unstable':
                fillstyle = 'none'
            elif cls == 'Left half-stable':
                fillstyle = 'left'
            elif cls == 'Right half-stable':
                fillstyle = 'right'
            else:
                continue
        else:
            fillstyle = 'full' if stable else 'none'
        plt.plot(fp, 0, marker='o', fillstyle=fillstyle, markersize=8, color='k')

    # the slope field
    if n_arrows > 0:
        skip = fp_resolution
        x_dot_ = x_dot[skip//2::skip]
        x_dot_ = np.sign(x_dot_)*np.sqrt(np.abs(x_dot_))
        x_ = x[skip//2::skip]
        q = ax.quiver(x_, np.zeros_like(x_), x_dot_, np.zeros_like(x_dot_), np.abs(x_dot_),
                    cmap='cool', pivot='mid', angles='xy', headwidth=3, headlength=1, headaxislength=1)
        ax.quiverkey(q, X=0.8, Y=1.03, U=2, label='dx/dt', labelpos='E')
    ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(*ax.get_ylim())

    ax.set_title(title)
    ax.set_xlabel('$' + x_label + '$')
    ax.set_ylabel('$\\dot{'+x_label+'}$')
    if grid:
        ax.grid()
    if show:
        plt.show()

def ODE_phase_2d(f, x0s=None, xlim=(-2,2), ylim=(-2,2), dims=(None, None), T=30, n_timesteps=6000, ax=None, x_arrows=20, y_arrows=20, figsize=None, x_label="x", y_label="y", title="Phase portrait", trajectory_width=1,
              fp_resolution=100, fp_filter_eps=2.5e-3, fp_distance_eps=1e-1, stability_method='jacobian', fp_stability_eps=1e-5, nullclines=False, nullclines_eps=5e-4):
    """
    Phase portrait of a first-order 2D ODE system

    Hints:
    - If there are multiple fixed points stacked on top of each other, try increasing fp_distance_eps
    - If the nullclines seem incomplete, try increasing `nullclines_eps` and/or `fp_resolution`
    - `stability_method = 'lyapunov'` is significantly slower and often needs a larger `fp_stability_eps` to work well, but is more reliable for non-hyperbolic fixed points

    Args:
    `f` (function):             The system of ODEs with at least two dimensions.
    `x0s` (list of iterables):  Initial conditions for the trajectories (number of dimensions must match the number of input to `f`)
    `xlim` (tuple):             The limits of the x axis
    `ylim` (tuple):             The limits of the y axis
    `dims` (tuple):             If the system has more than two dimensions, select which dimensions to plot by setting them to `None` and provide default values for the other dimensions.
    `T` (float):                The time to simulate the trajectories and check the stability of the fixed points
    `n_timesteps` (int):        The number of timesteps to simulate the trajectories in `[0, T]`
    `ax` (matplotlib axis):     Optional axis to plot the phase portrait
    `x_arrows` (int):           The number of arrows on the vector field in the x direction
    `y_arrows` (int):           The number of arrows on the vector field in the y direction
    `figsize` (tuple):          A tuple of two floats. If `None`, the size will be adjusted to keep the aspect ratio of `xlim` and `ylim`.
    `x_label` (str):            The label of the x axis
    `y_label` (str):            The label of the y axis
    `title` (str):              The title of the plot
    `trajectory_width` (float): The width of the trajectories
    `fp_resolution` (int):      The resolution of the grid (`fp_resolution*x_arrows x fp_resolution*y_arrows`) in which to look for fixed points and nullclines. Higher values will make the execution much slower.
    `fp_filter_eps` (float):    The maximum `|f(x^*, y^*)|` for which `(x^*, y^*)` is considered a fixed point
    `fp_distance_eps` (float):  The maximum distance between fixed points for them to be considered the same.
    `stability_method` (str):   The method to check for stability: 'jacobian' or 'lyapunov'.
    `fp_stability_eps` (float): The precision when checking for stability. If method is `jacobian`, it controls the precision of the finite differences and for method `lyapunov` the distance of the test particle.
    `nullclines` (bool):        Whether to plot the nullclines. If True, the nullclines will be plotted in red (x) and black (y)
    `nullclines_eps` (float):   The maximum allowed absolute value. Higher values will make the nullclines clearer visible, but might smear out on plateaus
    """

    def get_nullclines(dot, eps, xmin, dx, ymin, dy):
        rows, cols = np.where(np.abs(dot) < eps)
        ps = []
        for r,c in zip(rows, cols):
            p = xmin + c*dx, ymin + r*dy
            ps.append(p)

        return np.array(ps)

    x_min, x_max = xlim
    y_min, y_max = ylim
    dt = T/n_timesteps
    if x_arrows is None or x_arrows < 1:
        x_arrows = 0
        dx = (x_max - x_min)/fp_resolution
    else:
        dx = (x_max - x_min)/(fp_resolution*x_arrows)
    if y_arrows is None or y_arrows < 1:
        y_arrows = 0
        dy = (y_max - y_min)/fp_resolution
    else:
        dy = (y_max - y_min)/(fp_resolution*y_arrows)
    x, y = np.meshgrid(np.arange(x_min, x_max, dx), np.arange(y_min, y_max, dy))

    xdim = dims.index(None)
    ydim = dims.index(None, xdim+1)
    if len(dims) > 0:
        # place x,y, where dims is None and pick the output of f accordingly
        def f_2d(x,y):
            args = list(dims)
            args[xdim] = x
            args[ydim] = y
            return f(*args)
    else:
        f_2d = f
    dots = f_2d(x,y)
    x_dot, y_dot = dots[xdim], dots[ydim]

    # the slope field
    ax_orig = ax
    if ax is None:
        if figsize is None:
            # figsize reflects the aspect ratio of xlim and ylim, anker the smaller axis
            anker = 5
            aspect = (x_max - x_min)/(y_max - y_min)
            if aspect < 1:
                figsize = (anker, anker/aspect)
            else:
                figsize = (anker*aspect, anker)
        fig, ax = plt.subplots(figsize=figsize)
    if x_arrows > 0 and y_arrows > 0:
        skip = fp_resolution
        x_dot_ = x_dot[::skip, ::skip]
        y_dot_ = y_dot[::skip, ::skip]
        x_ = x[::skip, ::skip]
        y_ = y[::skip, ::skip]
        q = ax.quiver(x_, y_, x_dot_, y_dot_, np.sqrt(np.square(x_dot_) + np.square(y_dot_)),
                    cmap='jet', pivot='mid', angles='xy')
        ax.quiverkey(q, X=0.8, Y=1.03, U=4, label='dy/dx', labelpos='E')
        #ax.set_xlim(*ax.get_xlim())
        #ax.set_ylim(*ax.get_ylim())
        x_arrow = (x_min - x_max)/x_arrows
        y_arrow = (y_min - y_max)/y_arrows
        ax.set_xlim(x_min + .5*x_arrow, x_max + .5*x_arrow)
        ax.set_ylim(y_min + .5*y_arrow, y_max + .5*y_arrow)

    # nullclines
    if nullclines:
        if fp_resolution is None or fp_resolution < 1:
            raise ValueError("fp_resolution must be >= 1 to plot nullclines")
        if isinstance(nullclines_eps, (int, float)):
            nullclines_eps = 2*[nullclines_eps]
        elif len(nullclines_eps) != 2:
            raise ValueError("nullclines_eps must be a float or a tuple of three floats")
        for i, dot, c, nc_eps in zip(['x', 'y'], [x_dot, y_dot], ['r', 'k'], nullclines_eps):
            if isinstance(nullclines, str):
                if nullclines != i:
                    continue
            p = get_nullclines(dot, nc_eps, x_min, dx, y_min, dy)
            if len(p) > 0:
                ax.scatter(*p.T, linewidths=0, s=.5, color=c)

    # trajectories
    if x0s is not None:
        for x0 in x0s:
            res = simulate(f, x0, T, dt)
            x, y = res[xdim], res[ydim]
            x0x, x0y = x0[xdim], x0[ydim]
            ax.plot(x,y, linewidth=trajectory_width)
            ax.scatter(x0x, x0y, marker="x")

    # fixed points
    fps = find_fixed_points(f_2d, (x_dot, y_dot),(x_min, y_min), (dx,dy), fp_filter_eps, fp_distance_eps)
    for fp in fps:
        # use the default values for the other dimensions
        _fp = list(dims)
        _fp[xdim] = fp[0]
        _fp[ydim] = fp[1]
        if stability_method == 'jacobian':
            clss, stable = classify_fixed_point(f, _fp, fp_stability_eps, verbose=True)
            if clss == 'Unkown' or 'Unknown' in clss:
                continue
        elif stability_method == 'lyapunov':
            stable = is_stable(f, _fp, T, dt, fp_stability_eps, verbose=True)
        else:
            raise ValueError(f"stability_method must be 'lyapunov' or 'jacobian', not {stability_method}")

        if stable:
            ax.scatter(*fp, facecolors='k', edgecolors='k')
        else:
            ax.scatter(*fp, facecolors='none', edgecolors='k')

    ax.set_title(title)
    ax.set_xlabel('$' + x_label + '$')
    ax.set_ylabel('$' + y_label + '$')
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ax_orig is None:
        plt.show()

def ODE_phase_2d_polar(f, polar0s=None, x0s=None, rlim=2, **args):
    """
    Phase portrait of a first-order 2D ODE system in polar coordinates.

    Args:
    `f` (function):             The system of ODEs. Must take the two polar coordinates, `r` and `theta`, and return the derivatives `r_dot` and `theta_dot`
    `polar0s` (list of tuples): Initial conditions for the trajectories in polar coordinates
    `x0s` (list of tuples):     Initial conditions for the trajectories in cartesian coordinates
    `rlim` (float):             The maximum radius of the phase portrait
    `**args`:                   See arguments for `ODE_phase_2d`
    """
    def to_polar(x,y):
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y,x)
        return r, theta

    def to_cartesian(r, theta):
        return r*np.cos(theta), r*np.sin(theta)

    def f_cartesian(x,y):
        r, theta = to_polar(x,y)
        r_dot, theta_dot = f(r, theta)

        # see p. 153
        x_dot = r_dot * np.cos(theta) - r*theta_dot * np.sin(theta)
        y_dot = r_dot * np.sin(theta) + r*theta_dot * np.cos(theta)
#         y_dot = (r_dot * y + theta_dot * r * x)/ (r+1e-20)
#         x_dot = (r*r_dot - y*y_dot)/ (x+1e-20)
        return x_dot, y_dot

    if polar0s is not None:
        if x0s is None:
            x0s = []
        for r, theta in polar0s:
            x0 = to_cartesian(r, theta)
            x0s.append(x0)

    return ODE_phase_2d(f_cartesian, x0s, xlim=(-rlim, rlim), ylim=(-rlim, rlim), **args)

def ODE_phase_3d(f, x0s=None, xlim=(-2,2), ylim=(-2,2), zlim=(-2,2), dims=(None, None, None), T=30, n_timesteps=6000, x_label="x", y_label="y", z_label="z", title="Phase portrait", ax=None,
                fp_resolution=100, fp_filter_eps=2.5e-3, fp_distance_eps=1e-1, stability_method='jacobian', fp_stability_eps=1e-5, nullclines=False, nullclines_eps=5e-4, linewidth=.2):
    """
    Phase portrait of a first-order 3D ODE system

    Hints:
    - If there are multiple fixed points stacked on top of each other, try increasing `fp_distance_eps`

    Args:
    `f` (function):             The system of ODEs with at least three dimensions.
    `x0s` (list of iterables):  Initial conditions for the trajectories (number of dimensions must match the number of input to `f`)
    `xlim` (tuple):             The limits of the x axis
    `ylim` (tuple):             The limits of the y axis
    `zlim` (tuple):             The limits of the z axis
    `dims` (tuple):             If the system has more than three dimensions, select which dimensions to plot by setting them to `None` and provide default values for the other dimensions.
    `T` (float):                The time to simulate the trajectories and check the stability of the fixed points
    `n_timesteps` (int):        The number of timesteps to simulate the trajectories in `[0, T]`
    `x_label` (str):            The label of the x axis
    `y_label` (str):            The label of the y axis
    `z_label` (str):            The label of the z axis
    `title` (str):              The title of the plot
    `ax` (matplotlib axis):     Optional axis to plot the phase portrait
    `fp_resolution` (int):      The resolution of the grid in which to look for fixed points and plot the slope field
    `fp_filter_eps` (float):    The maximum `|f(x^*, y^*, z^*)|` for which `(x^*, y^*, z^*)` is considered a fixed point
    `fp_distance_eps` (float):  The maximum distance between fixed points for them to be considered the same.
    `stability_method` (str):   The method to check for stability: 'jacobian' or 'lyapunov'.
    `fp_stability_eps` (float): The precision when checking for stability. If method is `jacobian`, it controls the precision of the finite differences and for method `lyapunov` the distance of the test particle.
    `nullclines` (bool):        Whether to plot the nullclines. If True, the nullclines will be plotted in red (x), black (y), and green (z)
    `nullclines_eps` (float):   The maximum allowed absolute value. Higher values will make the nullclines clearer visible, but might smear out on plateaus
    `linewidth` (float):        The width of the trajectories
    """
    def get_nullclines(dot, eps, xmin, dx, ymin, dy, zmin, dz):
        rows, cols, depths = np.where(np.abs(dot) < eps)
        ps = []
        for r,c,d in zip(rows, cols, depths):
            p = xmin + c*dx, ymin + r*dy, zmin + d*dz
            ps.append(p)

        return np.array(ps)

    dt = T/n_timesteps
    x_min, x_max = xlim
    y_min, y_max = ylim
    z_min, z_max = zlim

    if fp_resolution >= 1:
        dx = (x_max - x_min)/(fp_resolution)
        dy = (y_max - y_min)/(fp_resolution)
        dz = (z_max - z_min)/(fp_resolution)
        x, y, z = np.meshgrid(np.arange(x_min, x_max, dx), np.arange(y_min, y_max, dy), np.arange(z_min, z_max, dz))

        xdim = dims.index(None)
        ydim = dims.index(None, xdim+1)
        zdim = dims.index(None, ydim+1)
        if len(dims) > 3:
            # place x,y, where dims is None and pick the output of f accordingly
            def f_3d(x,y,z):
                args = list(dims)
                args[xdim] = x
                args[ydim] = y
                args[zdim] = z
                return f(*args)
        else:
            f_3d = f
        x_dot, y_dot, z_dot = f_3d(x,y,z)

    # no slope field for 3D
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # nullclines
    if nullclines:
        if fp_resolution is None or fp_resolution < 1:
            raise ValueError("fp_resolution must be >= 1 to plot nullclines")
        if isinstance(nullclines_eps, (int, float)):
            nullclines_eps = 3*[nullclines_eps]
        elif len(nullclines_eps) != 3:
            raise ValueError("nullclines_eps must be a float or a tuple of three floats")
        for i, dot, c, nc_eps in zip(['x', 'y', 'z'], [x_dot, y_dot, z_dot], ['r', 'k', 'g'], nullclines_eps):
            if isinstance(nullclines, str):
                if nullclines != i:
                    continue
            p = get_nullclines(dot, nc_eps, x_min, dx, y_min, dy, z_min, dz)
            if len(p) > 0:
                ax.scatter(*p.T, linewidths=0, s=.5, color=c)

    # trajectories
    if x0s is not None:
        for x0 in x0s:
            x, y, z = simulate(f, x0, T, dt)[0:3]
            ax.plot(x, y, z, linewidth=linewidth)
            ax.scatter(*x0, marker="x")

    # fixed points
    if fp_resolution >= 1:
        fps = find_fixed_points(f_3d, [x_dot, y_dot, z_dot], [x_min, y_min, z_min], [dx, dy, dz], fp_filter_eps, fp_distance_eps)
        for fp in fps:
            _fp = list(dims)
            _fp[xdim] = fp[0]
            _fp[ydim] = fp[1]
            _fp[zdim] = fp[2]
            if stability_method == 'jacobian':
                clss, stable = classify_fixed_point(f, _fp, fp_stability_eps, verbose=True)
                if 'Unknown' in clss:
                    continue
            elif stability_method == 'lyapunov':
                stable = is_stable(f, _fp, T, dt, fp_stability_eps, verbose=True)
            else:
                raise ValueError(f"stability_method must be 'lyapunov' or 'jacobian', not {stability_method}")

            if stable:
                ax.scatter(*fp, color='k')
            else:
                ax.scatter(*fp, color='r')

    ax.set_title(title)
    ax.set_xlabel('$' + x_label + '$')
    ax.set_ylabel('$' + y_label + '$')
    ax.set_zlabel('$' + z_label + '$')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    plt.show()


###########################
### Bifurcation diagram ###
###########################

def get_roots(f, x0s, filter_eps=1e-5, distance_eps=1e-5):
    """
    Find the roots of a function (i.e. the fixed points of an ODE system) using `fsolve`, starting from multiple initial conditions `x0s`
    and filtering them by `filter_eps` (actually being roots) and `distance_eps` (mean out roots that are too close to each other).
    """
    roots_ = []
    for x0 in x0s:
        cand = fsolve(f, x0)
        if np.linalg.norm(f(cand)) < filter_eps:
            roots_.append(cand)

    # mean out roots that are too close to each other
    roots_lists = []
    for root in roots_:
        found = False
        for roots_list in roots_lists:
            if np.mean(np.linalg.norm(np.array(roots_list) - np.array(root), axis=1)) < distance_eps:
                roots_list.append(root)
                found = True
                break
        if not found:
            roots_lists.append([root])
    roots = []
    for roots_list in roots_lists:
        roots.append(np.mean(roots_list, axis=0))

    return roots

def bifurcation_diagram_1d(f, x0s, r_range, r_res=200, fp_filter_eps=1e-5, fp_distance_eps=1e-2, stability_method='jacobian', fp_stability_eps=1e-5, dim=0, x_label='r', y_label='x', title='Bifurcation diagram'):
    """
    Bifurcation diagram of a 1D ODE system

    Args:
    `f` (function):             A function that take a parameter `r` and returns an ODE, i.e., a function taking `x` and return the first derivative `x_dot`
    `x0s` (list of floats):     Initial conditions for the trajectories, used to find the fixed points
    `r_range` (tuple):          The range of the parameter `r`
    `r_res` (float):            The step size of the parameter `r`
    `fp_filter_eps` (float):    The maximum `|f(x^*)|` for which `x^*` is considered a fixed point
    `fp_distance_eps` (float):  The maximum distance between fixed points for them to be considered the same.
    `stability_method` (str):   The method to check for stability: 'jacobian' or 'lyapunov'.
    `fp_stability_eps` (float): The precision when checking for stability. If method is `jacobian`, it controls the precision of the finite differences and for method `lyapunov` the distance of the test particle.
    `dim` (int):                The dimension of the ODE system to plot against the parameter
    `x_label` (str):            The label of the x axis
    `y_label` (str):            The label of the y axis
    `title` (str):              The title of the plot
    """
    # get roots for each r
    dr = (r_range[1]-r_range[0])/r_res
    rs = np.arange(*r_range, dr)
    all_roots = []
    for r in rs:
        roots = get_roots(f(r), x0s, fp_filter_eps, fp_distance_eps)
        # print(r, roots)
        all_roots.append(roots)

    # stability analysis
    stabilities = []
    if fp_stability_eps is not None:
        for r, roots in zip(rs, all_roots):
            if stability_method == 'jacobian':
                stabilities.append([classify_fixed_point(f(r), root, eps=fp_stability_eps, verbose=False) for root in roots])
            elif stability_method == 'lyapunov':
                stabilities.append([is_stable(f(r), root, eps=fp_stability_eps, verbose=False) for root in roots])
            else:
                raise ValueError(f"method must be 'jacobian' or 'lyapunov', not {stability_method}")

    plt.figure(figsize=(8, 5))
    # scatter stable roots as black dots and unstable roots as red dots
    for r, roots, stables in zip(rs, all_roots, stabilities):
        for root, (clss, stable) in zip(roots, stables):
            if clss == 'Unknown' or 'Unknown' in clss:
                continue
            color = 'k' if stable else 'r'
            if not isinstance(root, (int, float)):
                root = root[dim]
            plt.scatter(r, root, color=color, s=2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(r_range)
    plt.grid()
    plt.show()

    return rs, all_roots, stabilities

########################
## Stability diagram ###
########################

def stability_diagram(f, x0s, a_range, b_range, res=100, fp_filter_eps=1e-5, fp_distance_eps=1e-5, fp_stability_eps=1e-5, kind='log', x_label='a', y_label='b', title='Stability diagram'):
    """
    Stability diagram for two parameters of an n-dimensional first-order ODE system, using linear stability analysis.

    Args:
    `f` (function):             A function that takes two arguments, `a` and `b`, and returns a first-order ODE system, a function that takes one argument, `x`, and returns the first derivative `x_dot`
    `x0s` (list of floats):     Initial conditions for the trajectories, used to find the fixed points
    `a_range` (tuple):          The range of the parameter `a`
    `b_range` (tuple):          The range of the parameter `b`
    `res` (float):              The resolution in the parameters `a` and `b`
    `fp_filter_eps` (float):    The maximum `|f(x^*)|` for which `x^*` is considered a fixed point
    `fp_distance_eps` (float):  The maximum distance between fixed points for them to be considered the same.
    `fp_stability_eps` (float): The precision of the finite differences.
    `kind` (str or list):       The kind of plot to show. Can be 'log', 'roots', 'real', '+log', 'dis', or 'all'
    `x_label` (str):            The label of the x axis
    `y_label` (str):            The label of the y axis
    `title` (str):              The title of the plot
    """
    # get roots for each combination of a and b
    da = (a_range[1]-a_range[0])/res
    db = (b_range[1]-b_range[0])/res
    a_s = np.arange(*a_range, da)
    b_s = np.arange(*b_range, db)
    all_roots = []
    for a, b in tq(product(a_s, b_s), desc='Calculating roots', total=len(a_s)*len(b_s)):
        roots = get_roots(f(a, b), x0s, fp_filter_eps, fp_distance_eps)
        all_roots.append(roots)

    if isinstance(kind, str):
        kind = [kind]
    all = ['roots', 'real', 'log', 'dis', 'logdis']
    if 'all' in kind:
        for k in all:
            if k not in kind:
                kind.insert(kind.index('all'), k)
        while 'all' in kind:
            kind.remove('all')

    for k in kind:
        allowed = all + ['+log']
        if not k in allowed:
            raise ValueError(f"kind must be '" + "', '".join(allowed) + "', or 'all', not {kind}")

    # find whether any root is close to a bifurcation point (i.e. f(x)/dx is close to 0)
    if any(k != 'roots' for k in kind):
        bifurcations = np.zeros((len(b_s), len(a_s)))
        discriminants = np.zeros((len(b_s), len(a_s)))
        for (i, j), roots in tq(zip(product(range(len(a_s)), range(len(b_s))), all_roots), desc='Finding bifurcations', total=len(all_roots)):
            a = a_s[i]
            b = b_s[j]
            dfs = []
            diss = []
            fab = f(a, b)
            for root in roots:
                # find df/dx at the root using finite differences
                eps = fp_stability_eps
                if len(root) == 1:
                    # derivative of the function
                    dfu = fab(root + eps)
                    dfl = fab(root - eps)
                    if dfu is None or dfl is None:
                        continue
                    df = (dfu - dfl)/(2*eps)
                    dfs.append(float(df))
                else:
                    # smallest absolute eigenvalue of the Jacobian matrix
                    dims = len(root)
                    J = np.zeros((dims, dims))
                    for k in range(dims):
                        # partial derivative of f with respect to x_k at the root
                        dfabu = np.array(fab(root + eps*np.eye(dims)[:,k]))
                        dfabl = np.array(fab(root - eps*np.eye(dims)[:,k]))
                        J[:,k] = (dfabu - dfabl)/(2*eps)
                    if np.any(np.isnan(J)):
                        continue
                    if 'dis' in kind or 'logdis' in kind:
                        tr = np.trace(J)
                        det = np.linalg.det(J)
                        dis = tr**2 - 4*det
                        diss.append(dis)
                        # print(i, j, a, b, tr, det, dis)
                    evs = np.linalg.eigvals(J).real
                    dfs.append(evs[np.argmin(np.abs(evs))])
                    # print(i, j, a, b, evs)
            # print(i, j, a, b, dfs)
            v = dfs[np.argmin(np.abs(dfs))] if len(dfs) > 0 else None
            bifurcations[len(b_s)-1-j, i] = v #if np.abs(v) < 3e-1 else 0
            if 'dis' in kind or 'logdis' in kind:
                v = diss[np.argmin(np.abs(diss))] if len(diss) > 0 else None
                discriminants[len(b_s)-1-j, i] = v

    # plot
    for k in kind:
        plt.figure(figsize=(10, 6))
        if k == 'log':
            logbiabs = -np.log(np.abs(bifurcations))
            # replace nan with 0 -> show as black (no bifurcation)
            logbiabs[np.isnan(logbiabs)] = 0
            # vmax = np.nanmax(logbiabs[logbiabs != np.inf])
            plt.imshow(logbiabs, extent=(*a_range, *b_range), aspect='auto', cmap='hot')
            plt.colorbar(label='-log(abs(df/dx))')
        elif k == '+log':
            logbiabs = np.log(np.abs(bifurcations))
            vmin = np.nanmin(logbiabs[logbiabs != -np.inf])
            logbiabs[logbiabs == -np.inf] = vmin
            plt.imshow(logbiabs, extent=(*a_range, *b_range), aspect='auto', cmap='hot')
            plt.colorbar(label='log(abs(df/dx))')
        elif k == 'roots':
            n_roots = np.zeros((len(b_s), len(a_s)))
            for (i, j), roots in zip(product(range(len(a_s)), range(len(b_s))), all_roots):
                n_roots[len(b_s)-1-j, i] = len(roots)
            plt.imshow(n_roots, extent=(*a_range, *b_range), aspect='auto', cmap='hot', vmin=0)
            plt.colorbar(label='Number of roots')
        elif k == 'real':
            plt.imshow(bifurcations, extent=(*a_range, *b_range), aspect='auto', cmap='seismic', vmin=-np.max(np.abs(bifurcations)), vmax=np.max(np.abs(bifurcations)))
            plt.colorbar(label='df/dx')
        elif k == 'dis':
            plt.imshow(discriminants, extent=(*a_range, *b_range), aspect='auto', cmap='seismic', vmin=-np.max(np.abs(discriminants)), vmax=np.max(np.abs(discriminants)))
            plt.colorbar(label='Discriminant')
        elif k == 'logdis':
            # normalize discriminants
            logbiabs = -np.log(np.abs(discriminants))
            # replace nan with 0 -> show as black (no transition)
            logbiabs[np.isnan(logbiabs)] = 0
            # vmax = np.nanmax(logbiabs[logbiabs != np.inf])
            plt.imshow(logbiabs, extent=(*a_range, *b_range), aspect='auto', cmap='hot')
            plt.colorbar(label='-log(abs(discriminant))')

        plt.gca().spines[['top', 'right', 'bottom', 'left']].set_visible(False)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim(a_range)
        plt.ylim(b_range)
        plt.show()

############
### Misc ###
############

def plt_diagonal(ax=None, fmt='k-', linewidth=.5):
    if ax is None:
        ax = plt.gca()
    extent = ax.axis()
    start = extent[0] if np.abs(extent[0]) > np.abs(extent[2]) else extent[2]
    end = extent[1] if np.abs(extent[1]) < np.abs(extent[3]) else extent[3]
    ax.plot([start, end], [start, end], fmt, linewidth=linewidth)
    return ax

def lorenz_map(f, x0, dim=0, T=1000, n_timesteps=100000, figsize=(6,7), cmap='plasma', s=5, kind='scatter'):
    from scipy.signal import find_peaks

    res = simulate(f, x0, T, T/n_timesteps)
    x, t = res[:-1][dim], res[-1]
    peaks, _ = find_peaks(x)
    if len(peaks) == 0:
        raise ValueError('No peaks found. Does the system have periodic behavior?')
    if len(peaks) < 2:
        raise ValueError('Not enough peaks found. Consider increasing `T` or `n_timesteps`.')
    fixed_point = peaks[:-1][np.argmin(np.abs(x[peaks][:-1] - x[peaks][1:]))]
    var = f'xyz'[dim] if dim < 3 else 'x'
    print(f"Fixed point at {var}_n ≈ {x[fixed_point]} {var}_{{n+1}} ≈ {x[fixed_point+1]}")

    plt.figure(figsize=figsize)
    if kind not in ['scatter', 'coweb', 'both']:
        raise ValueError('kind must be one of "scatter", "coweb", "both"')
    if kind == 'scatter' or kind == 'both':
        plt.scatter(x[peaks][:-1], x[peaks][1:], c=t[peaks][1:], cmap=cmap, s=s)
    if kind == 'coweb' or kind == 'both':
        xs = [x[peaks][0]]
        ys = [0]
        for i in range(len(peaks)-1):
            xs.extend([x[peaks][i], x[peaks][i+1]])
            ys.extend([x[peaks][i+1], x[peaks][i+1]])
        plt.plot(xs, ys, c='b', linewidth=.5)
    plt_diagonal()
    plt.xlabel(f'${var}_n$')
    plt.ylabel('$'+var+'_{n+1}$')
    if kind == 'scatter' or kind == 'both':
        plt.colorbar(label='t', orientation='horizontal', aspect=50, pad=0.1)
    return x[peaks], t[peaks]

def to_lorenz_map(f, dim=0, T=2, n_timesteps=200):
    """
    Returns a iterative map for the ODE `f`, returnings the next peak in the `dim`-th dimension after a given `x0`.
    """
    from scipy.signal import find_peaks
    def lorenz_map(x):
        res = simulate(f, x, T, T/n_timesteps)
        x, t = res[:-1][dim], res[-1]
        peaks, _ = find_peaks(x)
        if len(peaks) > 10:
            print(f"Warning: {len(peaks)} peaks found. Consider decreasing `T` or increasing `n_timesteps`.")
        # return the next peak
        return x[peaks[0]]
    return lorenz_map

######################
### Iterative maps ###
######################

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
    x0s = np.array(x0s)
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
    fp = np.array(fp)
    # Calculate the Jacobian matrix
    dims = len(fp)
    J = np.zeros((dims, dims))
    for i in range(dims):
        eta = eps*np.eye(dims)[:,i]
        J[:, i] = (f(fp + eta) - f(fp - eta))/(2*eps)
    # Calculate the eigenvalues
    eigvals = np.linalg.eigvals(J)
    # Check the stability
    stable = np.all(np.abs(eigvals) < 1)
    classes = []
    for e in eigvals:
        if np.abs(e) < 1e-6:
            classes.append('superstable')
        elif np.abs(e - 1) < 1e-6 or np.abs(e + 1) < 1e-6:
            classes.append('nonlinear')
        elif np.abs(e) < 1:
            classes.append('stable')
        else:
            classes.append('unstable')
    return {'multipliers': eigvals, 'cls': classes, 'is_stable': stable}, stable

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
            except:
                break
            xs.extend([x, x1])
            ys.extend([x1, x1])
        # print(conv)
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
            plt.plot(fp, fp, 'ro', markersize=5)
        else:
            # empty circle (unstable fixed point)
            plt.plot(fp, fp, 'wo', markersize=5, markeredgecolor='r')

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
        show (bool):     Whether to call `plt.show()`.
        ppi (int):       The resolution of the plot in pixels per inch.
        **plt_kwargs:    Keyword arguments passed to `plt.imshow`.

    Returns
        ndarray: The final values of `z`.
        ndarray: The number of iterations needed for convergence.
    """

    def _fractal(f, z0, max_iters, eps=1e-6):
        z = np.array(z0, copy=False)
        error = np.inf
        iters = np.zeros_like(z, dtype='i4')
        for it in tq(range(1,max_iters+1)):
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
        return z, iters

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
    z, iters = _fractal(f, mesh, max_iters, eps)

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
        if show:
            plt.show()
        if save_fig is not None:
            filename = save_fig + f'_{name}.png'
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            print(f"Saved to {filename}")

    if kind == 'it' or kind == 'both':
        _plot(iters, 'it')
    if kind == 'z' or kind == 'both':
        img = colorize_complex(z)
        _plot(img, 'z')

    return z, iters