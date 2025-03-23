import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm as tq

from .data import moving_avg
from .prob import smooth
from .systems import fractal, simulate, Subgrid, GameOfLife
from .plot import imshow_dynamic, plot_dynamic

def climateHockey():
    import pandas as pd

    # download the data from https://ourworldindata.org/explorers/climate-change
    # get data
    try:
        df = pd.read_csv('climate-change.csv', parse_dates=["Date"])
    except FileNotFoundError as e:
        print("Download the data from https://ourworldindata.org/explorers/climate-change and save the csv in the current directory")
        return
    temp = df[df["Entity"] == "Northern Hemisphere"].set_index("Date")["Temperature anomaly"]
    # calculate smoothing
    y = smooth(temp)
    smoothed = pd.Series(y,index=temp.index)
    # xticks for plotting (data starts at 1880-01-15)
    idx = pd.date_range(start = temp.index[0], end = temp.index[-1], freq = "5Y") + pd.DateOffset(days=15,years=-1)
    # plot
    plt.figure(figsize=(10,6))
    temp.plot(rot=45, xticks=idx, alpha=0.3)
    smoothed.plot(color='red', label="Smoothed") # for some reason also changes xticks to year-only
    plt.plot(temp.index[15:-15], moving_avg(temp,31), label="31-day moving average")
    plt.legend()
    plt.grid()
    plt.title("Temperature as deviation from the 1951-1980 mean")
    plt.ylabel("Temperature anomaly")
    plt.margins(0.01) # axis limits closer to graph
    plt.show()

def fractal_heart():
    import warnings; warnings.filterwarnings("ignore")

    phi = np.exp(1j*1/4*2*np.pi)
    f = lambda it, z, z0: z**2 + 0.7 + z0*phi if it > 1 else (z*phi)**2 + 0.7 + z0*phi
    fractal(f, 100, 2000, (-1, 1), (-0.2, 1.8), show='open', save_fig='fractal_heart')

    f = lambda it, z, z0: z**2 + 0.7 + z0
    fractal(f, 4000, 2000, (-0.79804, -0.79682), (-0.65403, -0.65281), show='open', save_fig='fractal_heart_zoomed')

def mandelbrot(res=2000, iters=120, xlim=(-2, 1), ylim=(-1.5, 1.5), cmap='default', dynamic=False):
    if cmap == 'default':
        from matplotlib.colors import LinearSegmentedColormap
        xlim, ylim = (-2, 1), (-1.5, 1.5)
        colors = ["black", "black", "blue", "lightblue", "white", "yellow", "red", "black"]
        nodes = [0.0, 0.02, 0.25, 0.35, 0.4, 0.7, 0.9, 1.0]
        cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))
    import warnings; warnings.filterwarnings("ignore")
    f = lambda it, z, z0: z**2 + z0
    if dynamic:
        gen = (g[1] for g in fractal(f, iters, res, xlim, ylim, show='gen', cmap=cmap))
        def cb(x, im, fig):
            im.autoscale()
            return x
        return imshow_dynamic(gen, figsize=(4,4), sleep=20, cb=cb)
    else:
        fractal(f, iters, res, xlim, ylim, show='open', save_fig='mandelbrot', cmap=cmap)

def lorenz(T=5000, sleep=5):
    def lorenz(x,y,z, sigma=10, rho=28, beta=8/3):
        return sigma*(y-x), x*(rho-z)-y, x*y-beta*z

    def lorenz_gen(T, dt=0.01):
        x,y,z,_ = simulate(lorenz, 10*np.random.rand(3), T, dt)
        for t in tq(range(T)):
            yield x[t],z[t]

    return plot_dynamic(lorenz_gen(T), (5,5), xlim=(-30,30), ylim=(-5,55), linewidth=0.3, sleep=sleep, dot='r.')

def game_of_life(rules='conway', T=255, size=256, sleep=50, skip=0, **kwargs):
    """
    Generalized Game of Life simulation.
    Use a matplotlib backend suitable for interactive plotting, e.g. `%matplotlib qt`.

    Parameters:
      rules (str): The ruleset. The following are available:
        'conway', 'highlife', 'rule30', 'rule90', 'sierpinski', 'ising', or 'schelling'.
      T (int): Number of iterations.
      size (int or tuple): Size of the grid. Rules 'rule30' and 'rule90' require an int, the others allow either an int or a tuple of two ints (rows, cols).
      sleep (int): Time in ms to sleep between iterations.
      skip (int): Number of iterations to skip between updates.
      **kwargs: Rule-specific parameters:
        - 'conway', 'highlife': seed_frac (default 0.1)
        - 'ising': temperature (default 2.27), update_frac (default 0.1), p (default 0.75)
        - 'schelling': tolerance (default 0), base_move_prob (default 0.01), p (default [0.45, 0.45])
    """
    conway_kernel = [
        [1,1,1],
        [1,0,1],
        [1,1,1]
    ]
    def init_random(p):
        return lambda rows, cols: np.random.choice([0,1], size=(rows, cols), p=[1-p, p])

    def init_seeds(n_seeds):
        def _init(rows, cols):
            grid = np.zeros((rows, cols), dtype='i1')
            for _ in range(n_seeds):
                i, j = np.random.randint(1,rows-2), np.random.randint(1,cols-2)
                grid[i:i+3,j:j+3] = np.random.randint(2, size=(3,3), dtype='i1')
            return grid
        return _init

    def init_dot(rows, cols):
        grid = np.zeros((rows, cols), dtype='i1')
        grid[-1, cols//2] = 1
        return grid

    if rules == 'conway':
        def f(t, grid):
            n = Subgrid(grid)(conway_kernel)
            survi =  grid & (n == 2) | (n == 3)  # Survival with 2 or 3 neighbors
            birth = ~grid & (n == 3)             # Birth with 3 neighbors
            return survi | birth
        seed_frac = 0.1
    elif rules == 'highlife':
        def f(t, grid):
            n = Subgrid(grid)(conway_kernel)
            survi =  grid & (n == 2) | (n == 3)  # Survival with 2 or 3 neighbors
            birth = ~grid & (n == 3) | (n == 6)  # Birth with 3 or 6 neighbors
            return survi | birth
        seed_frac = 0.01
    elif rules == 'rule30':
        def f(t, grid):
            s = Subgrid(grid)
            r1d = s.E ^ (s.C | s.W)
            return r1d | s.N
    elif rules == 'rule90' or rules == 'sierpinski':
        def f(t, grid):
            s = Subgrid(grid)
            r1d = s.W ^ s.E
            return r1d | s.N
    elif rules == 'ising':
        def ising2d(temperature, update_frac):
            def ising2d(t, grid):
                E = Subgrid(grid)([
                    [0,1,0],
                    [1,0,1],
                    [0,1,0]
                ])
                mask = np.random.rand(*grid.shape) < update_frac  # select a random subset of the grid to update
                dE = (E[mask] - 2) * (grid[mask]*2 - 1)  # rescale E to -2, 2 and grid to -1, 1
                flip = dE < 0
                if temperature > 0:
                    flip |= np.random.rand(dE.shape[0]) < np.exp(-(4/temperature) * dE)
                grid[mask] ^= flip
                return grid
            return ising2d

        def ising_cb(i, x, im, fig):
            fig.suptitle(f'{i}, magnetization: {2*np.mean(x)-1:.2f}')
            fig.canvas.draw()

        temperature = kwargs.get("temperature", 2.27)
        update_frac = kwargs.get("update_frac", 0.1)
        p = kwargs.get("p", 0.55)
        if isinstance(size, int):
            size = (size, size)

        return imshow_dynamic(GameOfLife(*size, T=T, rule=ising2d(temperature, update_frac), init=init_random(p)),
                    figsize=(6,6), sleep=sleep, skip=skip, cb=ising_cb)
    elif rules == 'schelling':
        def segregation(grid):
            n = Subgrid(grid)(conway_kernel)
            likeness = n * grid
            nonempty = likeness[grid != 0]
            if len(nonempty) == 0:
                return 0
            return np.mean(nonempty)

        def schelling(tolerance, base_move_prob=0):
            def _rule(t, grid):
                n = Subgrid(grid)(conway_kernel)
                new_grid = grid.copy()
                # people move away if they don't like it
                likeness = n * grid  # -1 wants more -1, 1 wants more 1, empty cells don't care
                p_move_away = (-np.tanh(likeness + tolerance) + 1)/2
                p_move_away = p_move_away*(1-base_move_prob) + base_move_prob
                move_away = np.random.rand(*grid.shape) < p_move_away
                new_grid[move_away] = 0
                # randomly distribute them over the empty cells
                n_moved = np.sum(move_away)
                emtpy_homes = grid == 0
                n_empty = np.sum(emtpy_homes)
                if n_moved > n_empty:
                    n_moved = n_empty  # don't add more people than empty spaces
                movers = np.zeros(n_empty)
                movers[:n_moved] = grid[move_away][:n_moved] #choice(np.array([-1, 1]), size=n_moved)
                if n_empty > n_moved:  # new people move in (doesn't change the emergent dynamics qualitatively, but speeds them up and makes the visuals look nicer)
                    movers[n_moved:] = np.random.choice([-1,0,1], size=n_empty-n_moved, p=[base_move_prob/2, 1-base_move_prob, base_move_prob/2])
                np.random.shuffle(movers)
                new_grid[emtpy_homes] = movers
                return new_grid
            return _rule

        def schelling_cb(t, x, im, fig):
            fig.suptitle(f"t={t}, segregation={segregation(x):.2f}")
            fig.canvas.draw()

        def schelling_init(p_1, p_2):
            assert p_1 + p_2 <= 1, "The sum of the two populations must be less than 1"
            p = [p_1, 1-p_1-p_2, p_2]
            def _init(rows, cols):
                return np.random.choice([-1,0,1], size=(rows, cols), p=p)
            return _init

        tolerance = kwargs.get("tolerance", 0)
        base_move_prob = kwargs.get("base_move_prob", 0.01)
        p = kwargs.get("p", [0.45, 0.45])
        if isinstance(size, int):
            size = (size, size)

        return imshow_dynamic(GameOfLife(*size, T=T, rule=schelling(tolerance, base_move_prob), init=schelling_init(*p)), figsize=(6,6),
                    sleep=sleep, skip=skip, cb=schelling_cb)
    else:
        raise ValueError(f"Invalid rules: {rules}. Choose from 'conway', 'highlife', 'rule30', 'rule90', 'sierpinski', or 'ising'.")

    if rules in ['conway', 'highlife']:
        seed_frac = kwargs.get("seed_frac", seed_frac)
        if isinstance(size, int):
            size = (size, size)
        return imshow_dynamic(GameOfLife(*size, T=T, rule=f, init=init_seeds(int(np.prod(size)*seed_frac))), figsize=(6,6), sleep=sleep, skip=skip)
    elif rules in ['rule30', 'rule90']:
        return imshow_dynamic(GameOfLife(size,2*size-1, T=T, rule=f, init=init_dot), figsize=(6,6), sleep=sleep, skip=skip)