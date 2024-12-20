import numpy as np

def simulate(f, x0, T, dt, method='RK45'):
    """
    Simulate an ODE system defined by `f` starting from initial condition `x0` for time `T` with time step `dt`.
    Returns the state trajectory and time points (*x, t).
    """
    from scipy.integrate import solve_ivp

    sol = solve_ivp(lambda t, state: f(*state), y0=x0, method=method, t_span=(0.0, T), dense_output=True)
    t = np.arange(0, T, dt)
    return *sol.sol(t), t

def get_roots(f, x0s, filter_eps=1e-5, distance_eps=1e-5):
    """
    Find the roots of a function (i.e. the fixed points of an ODE system) using `fsolve`, starting from multiple initial conditions `x0s`
    and filtering them by `filter_eps` (actually being roots) and `distance_eps` (mean out roots that are too close to each other).
    """
    from scipy.optimize import fsolve

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
            if np.mean(np.linalg.norm(np.array(roots_list) - np.asarray(root), axis=1)) < distance_eps:
                roots_list.append(root)
                found = True
                break
        if not found:
            roots_lists.append([root])
    roots = []
    for roots_list in roots_lists:
        roots.append(np.mean(roots_list, axis=0))

    return roots