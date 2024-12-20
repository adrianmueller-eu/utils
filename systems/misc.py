import warnings
import numpy as np
import matplotlib.pyplot as plt

from .ode import simulate

def potential(f, xmin=0, offset=0):
    from scipy.integrate import quad
    def _quad(x):
        return -np.array([quad(f, xmin, x)[0] + offset for x in x])
    return _quad

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
    Returns an iterative map for the ODE `f`, returnings the next peak in the `dim`-th dimension after a given `x0`.
    """
    from scipy.signal import find_peaks
    def lorenz_map(x):
        res = simulate(f, x, T, T/n_timesteps)
        x, t = res[:-1][dim], res[-1]
        peaks, _ = find_peaks(x)
        if len(peaks) > 10:
            warnings.warn(f"{len(peaks)} peaks found. Consider decreasing `T` or increasing `n_timesteps`.", stacklevel=2)
        elif len(peaks) < 2:
            warnings.warn('Insufficient peaks found. Try increasing `T` or `n_timesteps`.', stacklevel=2)
        elif len(peaks) == 0:
            raise ValueError('No peaks found. Does the system have periodic behavior?')
        # return the next peak
        return x[peaks[0]]
    return lorenz_map

def attractor_reconstruction(x, delta_1, delta_2=None, ax=None, show=True):
    # if delta_2 is not given, make a 2d reconstruction
    if delta_2 is None:
        x, y = x[:-delta_1], x[delta_1:]
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(x, y, '.', markersize=1)
        ax.set_aspect('equal')
        return x,y
    # if delta_2 is given, make a 3d reconstruction
    # delta_2 -= delta_1  # get the delay of the second coordinate over the first
    x,y,z = x[:-(delta_1 + delta_2)], x[delta_1:-delta_2], x[delta_1 + delta_2:]
    if ax is None:
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, '.', markersize=1)
    # equal axis ratio
    ax.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])
    if show:
        plt.show()
    return x,y,z