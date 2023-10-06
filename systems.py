import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp  # used for numerical integration
from itertools import product

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

def ODE_phase_2d(f, x0s=None, xlim=(-2,2), ylim=(-2,2), T=10, n_timesteps=100, ax=None, x_arrows=20, y_arrows=20,
              nullclines=False, nullclines_eps=5e-4, fp_resolution=100, fp_eps=2.5e-3, fp_stability_eps=1e-2):

    def get_nullclines(x_dot, y_dot, eps, xmin, dx, ymin, dy):
        rows, cols = np.where(np.logical_or(
            np.abs(x_dot) < eps, np.abs(y_dot) < eps
        ))
        ps = []
        for r,c in zip(rows, cols):
            p = xmin + c*dx, ymin + r*dy
            ps.append(p)

        return np.array(ps)

    def find_fixed_points(x_dot, y_dot, eps, xmin, dx, ymin, dy):
        row, col = np.where(np.logical_and(
            np.abs(x_dot) < eps, np.abs(y_dot) < eps
        ))
        #print(row.shape)
        fps = set()
        for r,c in zip(row, col):
            fp = [xmin + c*dx, ymin + r*dy]
            fp = np.round(fp, int(-np.log10(eps))-1)
            fps.add(tuple(fp))

        return fps

    def is_stable(f, fp, T, dt, eps=1e-2):
        fp = np.array(fp)
        noise = np.random.random(2)*eps
        #noise /= np.sum(noise**2) # normalize to length one = move to random direction
        x,y,_ = simulate(f, fp+noise, T, dt)
        res = [x[-1], y[-1]]
        error = np.sum((res - fp)**2)
        stable = error < eps # stable if it remains inside eps
        print(f"Fixed point {fp}: {stable} (eps: {error})") # init: {fp + noise}, res: {res},
        return stable

    x_min, x_max = xlim
    y_min, y_max = ylim
    dt = T/n_timesteps
    dx = (x_max - x_min)/(fp_resolution*x_arrows)
    dy = (y_max - y_min)/(fp_resolution*y_arrows)
    x, y = np.meshgrid(np.arange(x_min, x_max, dx), np.arange(y_min, y_max, dy))
    x_dot, y_dot = f(x,y)

    # the slope field
    if ax is None:
        fig, ax = plt.subplots()
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
    eps = nullclines_eps
    if nullclines:
        p = get_nullclines(x_dot, y_dot, eps, x_min, dx, y_min, dy)
        plt.scatter(*p.T, linewidths=0, s=.5, color="k")

    # trajectories
    if x0s is not None:
        for x0 in x0s:
            x, y = simulate(f, x0, T, dt)[0:2]
            plt.plot(x,y)
            plt.scatter(*x0, marker="x")

    # fixed points
    eps = fp_eps
    fps = find_fixed_points(x_dot, y_dot, eps, x_min, dx, y_min, dy)
    for fp in fps:
        if is_stable(f, fp, T, dt, fp_stability_eps):
            plt.scatter(*fp, facecolors='k', edgecolors='k')
        else:
            plt.scatter(*fp, facecolors='none', edgecolors='k')

    ax.set_title('Phase portrait')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
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

