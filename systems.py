import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp  # used for numerical integration
from scipy.optimize import fsolve

## This script contains functions to visualize and analyze dynamical systems
## Though, so far this only covers 1D and 2D first-order ODEs

############
### Flow ###
############

def simulate(f, x0, T, dt):
    sol = solve_ivp(lambda t, state: f(*state),
                    y0=x0, method='RK45', t_span=(0.0, T), dense_output=True)
    t = np.arange(0, T, dt)
    return *sol.sol(t), t

def ODE_flow_1d(f, x0s=None, x_limits=(-2,2), T=10, n_timesteps=100):
    x_min, x_max = x_limits
    dt = T/n_timesteps
    t, x = np.meshgrid(np.arange(0, T, T/20), np.arange(x_min, x_max, (x_max - x_min)/20))
    V = f(x)
    U = np.ones(V.shape)

    # slope field
    fig, ax = plt.subplots()
    q = ax.quiver(t, x, U, V, angles='xy')
    ax.quiverkey(q, X=0.8, Y=1.03, U=4, label='dx/dt', labelpos='E')
    ax.set_ylim(*ax.get_ylim())

    # trajectories
    if x0s is not None:
        for x0 in x0s:
            # convert x0 to a list if it is a scalar
            if isinstance(x0, (int, float)):
                x0 = [x0]
            s = reversed(simulate(f, x0, T, dt))
            plt.plot(*s)

    ax.set_xlim(-0.02*T, T)
    ax.set_title('Slope field')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()

def ODE_flow_2d(f, x0s=None, xlim=(-2,2), ylim=(-2,2), T=10, n_timesteps=100, ax=None):
    dt = T/n_timesteps

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    if x0s is not None:
        for x0 in x0s:
            x, y, t = simulate(f, x0, T, dt)
            ax.plot(t,x,y)
            ax.scatter(0, *x0, marker="x")

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
        print(f"Fixed point {fp}: {stable} (stability eps: {error} {s} {eps})") # init: {fp + noise}, res: {res},
    return stable

def ODE_phase_1d(f, x_limits=(-2,2), T=20, n_timesteps=200, ax=None, n_arrows=10, title="Phase portrait", x_label="x",
                 fp_resolution=1000, fp_filter_eps=2.5e-3, fp_distance_eps=1e-1, fp_stability_eps=1e-2):
    """
    Phase portrait of a first-order 1D ODE

    Hints:
    - If there are multiple fixed points stacked on top of each other, try increasing `fp_distance_eps`

    Args:
    `f` (function):             The ODE. Must take one argument, `x`, and return the first derivative `x_dot`
    `x_limits` (tuple):         The limits of the x-axis
    `T` (float):                The time to simulate the trajectories and check the stability of the fixed points
    `n_timesteps` (int):        The number of timesteps to simulate the trajectories in `[0, T]`
    `ax` (matplotlib axis):     Optional axis to plot the phase portrait
    `n_arrows` (int):           The number of arrows in the x axis
    `title` (str):              The title of the plot
    `x_label` (str):            The label of the x axis
    `fp_resolution` (int):      The resolution of the grid in which to look for fixed points and plot the slope field
    `fp_filter_eps` (float):    The maximum `|f(x^*)|` for which `x^*` is considered a fixed point
    `fp_distance_eps` (float):  The maximum distance between fixed points for them to be considered the same.
    `fp_stability_eps` (float): The distance of the test particle to check for Lyapunov stability
    """

    x_min, x_max = x_limits
    dt = T/n_timesteps
    dx = (x_max - x_min)/(fp_resolution*n_arrows)
    x = np.arange(x_min, x_max, dx)
    x_dot = f(x)

    # plot the function
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, x_dot)

    # fixed points
    def find_fixed_points(x_dot, x, filter_eps, distance_eps):
        fps = np.where(np.abs(x_dot) < filter_eps)
        # first, sort fps by closeness into lists
        fps_lists = []
        for i in fps[0]:
            found = False
            for j, fps_list in enumerate(fps_lists):
                # if np.abs(fps_list[0] - x[i]) < eps:
                if np.mean(np.abs(fps_list - x[i])) < distance_eps:
                    fps_list.append(x[i])
                    found = True
                    break
            if not found:
                fps_lists.append([x[i]])
        # then, calculate the mean of each list
        fps = []
        for fps_list in fps_lists:
            fps.append(np.mean(fps_list))
        return fps

    # print(fps)
    # round and set
    fps = find_fixed_points(x_dot, x, filter_eps=fp_filter_eps, distance_eps=fp_distance_eps)
    # print(fps)
    for fp in fps:
        if is_stable(f, [fp], T, dt, fp_stability_eps, verbose=True):
            plt.scatter(fp, 0, facecolors='k', edgecolors='k')
        else:
            plt.scatter(fp, 0, facecolors='none', edgecolors='r')

    # the slope field
    skip = fp_resolution
    x_dot_ = x_dot[skip//2::skip]
    x_dot_ = np.sign(x_dot_)*np.sqrt(np.abs(x_dot_))
    x_ = x[skip//2::skip]
    # print(x.shape, x_.shape, x_dot.shape, x_dot_.shape)
    q = ax.quiver(x_, np.zeros_like(x_), x_dot_, np.zeros_like(x_dot_), np.abs(x_dot_),
                  cmap='cool', pivot='mid', angles='xy', headwidth=3, headlength=1, headaxislength=1)
    ax.quiverkey(q, X=0.8, Y=1.03, U=2, label='dx/dt', labelpos='E')
    ax.set_xlim(*ax.get_xlim())
    ax.set_ylim(*ax.get_ylim())

    ax.set_title(title)
    ax.set_xlabel('$' + x_label + '$')
    ax.set_ylabel('$\dot{'+x_label+'}$')
    ax.grid()
    plt.show()

def ODE_phase_2d(f, x0s=None, xlim=(-2,2), ylim=(-2,2), T=30, n_timesteps=1000, ax=None, x_arrows=20, y_arrows=20, x_fig_size=6, title="Phase portrait", x_label="x", y_label="y",
              fp_resolution=100, fp_filter_eps=2.5e-3, fp_distance_eps=1e-1, fp_stability_eps=1e-2, nullclines=False, nullclines_eps=5e-4):
    """
    Phase portrait of a first-order 2D ODE system

    Hints:
    - If there are multiple fixed points stacked on top of each other, try increasing fp_distance_eps
    - If the nullclines seem incomplete, try increasing nullclines_eps and/or fp_resolution

    Args:
    `f` (function):             The system of ODEs. Must take two arguments, `x` and `y`, and return the derivatives `x_dot` and `y_dot`
    `x0s` (list of tuples):     Initial conditions for the trajectories
    `xlim` (tuple):             The limits of the x axis
    `ylim` (tuple):             The limits of the y axis
    `T` (float):                The time to simulate the trajectories and check the stability of the fixed points
    `n_timesteps` (int):        The number of timesteps to simulate the trajectories in `[0, T]`
    `ax` (matplotlib axis):     Optional axis to plot the phase portrait
    `x_arrows` (int):           The number of arrows on the vector field in the x direction
    `y_arrows` (int):           The number of arrows on the vector field in the y direction
    `x_fig_size` (float):       The size of the figure in the x axis. The size in the y axis will be adjusted to keep the aspect ratio of `xlim` and `ylim`
    `title` (str):              The title of the plot
    `x_label` (str):            The label of the x axis
    `y_label` (str):            The label of the y axis
    `fp_resolution` (int):      The resolution of the grid (`fp_resolution*x_arrows x fp_resolution*y_arrows`) in which to look for fixed points and nullclines. Higher values will make the execution much slower.
    `fp_filter_eps` (float):    The maximum `|f(x^*, y^*)|` for which `(x^*, y^*)` is considered a fixed point
    `fp_distance_eps` (float):  The maximum distance between fixed points for them to be considered the same.
    `fp_stability_eps` (float): The distance of the test particle to check for Lyapunov stability
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

    def find_fixed_points(x_dot, y_dot, filter_eps, distance_eps, xmin, dx, ymin, dy):
        row, col = np.where(np.logical_and(
            np.abs(x_dot) < filter_eps, np.abs(y_dot) < filter_eps
        ))

        # first, sort fps by closeness into lists
        fps_lists = []
        for r,c in zip(row, col):
            fp = [xmin + c*dx, ymin + r*dy]
            found = False
            for i, fps_list in enumerate(fps_lists):
                # if np.linalg.norm(np.array(fps_list[0]) - np.array(fp)) < eps:
                if np.mean(np.linalg.norm(np.array(fps_list) - np.array(fp), axis=1)) < distance_eps:
                    fps_list.append(fp)
                    found = True
                    break
            if not found:
                fps_lists.append([fp])
        # then, calculate the mean of each list
        fps = []
        for fps_list in fps_lists:
            fps.append(np.mean(fps_list, axis=0))

        return fps

    x_min, x_max = xlim
    y_min, y_max = ylim
    dt = T/n_timesteps
    dx = (x_max - x_min)/(fp_resolution*x_arrows)
    dy = (y_max - y_min)/(fp_resolution*y_arrows)
    x, y = np.meshgrid(np.arange(x_min, x_max, dx), np.arange(y_min, y_max, dy))
    x_dot, y_dot = f(x,y)

    # the slope field
    if ax is None:
        # figsize reflects the aspect ratio of xlim and ylim
        fig, ax = plt.subplots(figsize=(x_fig_size, x_fig_size*(y_max-y_min)/(x_max-x_min)))
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
        # x nullcline is black
        p = get_nullclines(x_dot, nullclines_eps, x_min, dx, y_min, dy)
        plt.scatter(*p.T, linewidths=0, s=.5, color="r")
        # y nullcline is gray
        p = get_nullclines(y_dot, nullclines_eps, x_min, dx, y_min, dy)
        plt.scatter(*p.T, linewidths=0, s=.5, color="k")

    # trajectories
    if x0s is not None:
        for x0 in x0s:
            x, y = simulate(f, x0, T, dt)[0:2]
            plt.plot(x,y)
            plt.scatter(*x0, marker="x")

    # fixed points
    fps = find_fixed_points(x_dot, y_dot, fp_filter_eps, fp_distance_eps, x_min, dx, y_min, dy)
    for fp in fps:
        if is_stable(f, fp, T, dt, fp_stability_eps, verbose=True):
            plt.scatter(*fp, facecolors='k', edgecolors='k')
        else:
            plt.scatter(*fp, facecolors='none', edgecolors='k')

    ax.set_title(title)
    ax.set_xlabel('$' + x_label + '$')
    ax.set_ylabel('$' + y_label + '$')
    plt.show()

def ODE_phase_2d_polar(f, polar0s=None, x0s=None, rlim=2, **args):
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

###########################
### Bifurcation diagram ###
###########################

def get_roots(f, x0s, prec=1e-5):
    roots = set()
    for x0 in x0s:
        cand = fsolve(f, x0)
        if np.linalg.norm(f(cand)) < prec:
            cand = np.round(cand, -int(np.log10(prec)))
            roots.add(tuple(cand))
    return list(roots)

def bifurcation_diagram_1d(f, x0s, r_range, dr=None, prec=1e-5, plot=True, stability=True, title='Bifurcation diagram'):
    # get roots for each r
    if dr is None:
        dr = (r_range[1]-r_range[0])/200
    rs = np.arange(*r_range, dr)
    all_roots = []
    for r in rs:
        roots = get_roots(lambda x: f(x, r), x0s, prec)
        # print(r, roots)
        all_roots.append(roots)

    # stability analysis
    if stability:
        stabilities = []
        for r, roots in zip(rs, all_roots):
            stabilities.append([is_stable(lambda x: f(x, r), root) for root in roots])

    if plot:
        plt.figure(figsize=(10, 6))
        # scatter stable roots as black dots and unstable roots as red dots
        for r, roots, stable in zip(rs, all_roots, stabilities):
            for root in roots:
                color = 'k' if stable[roots.index(root)] else 'r'
                plt.scatter(r, root, c=color, s=2)
        plt.title(title)
        plt.xlabel('r')
        plt.ylabel('x')
        plt.xlim(r_range)
        plt.grid()
    return rs, all_roots