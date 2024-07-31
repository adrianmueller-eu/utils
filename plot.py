import matplotlib.pyplot as plt
import numpy as np
import re
from .mathlib import is_complex, is_symmetric, int_sqrt
from .utils import *

def plot(x,y=None, fmt="-", figsize=(10,8), xlim=(None, None), ylim=(None, None), xlabel="", ylabel="", title="", xlog=False, ylog=False, grid=True, show=True, save_file=None, **pltargs):
    """Uses magic to create pretty plots."""

    # make it a bit intelligent
    if type(x) == tuple and len(x) == 2:
        title  = ylabel
        ylabel = xlabel
        if type(figsize) == str:
            xlabel = figsize
            if type(fmt) == tuple:
                figsize= fmt
                if type(y) == str:
                    fmt = y
        y = x[1]
        x = x[0]
    elif type(y) == str: # skip y
        title  = ylabel
        ylabel = xlabel
        if type(figsize) == str:
            xlabel = figsize
            if type(fmt) == tuple:
                figsize= fmt
        fmt=y
        y=None
    elif type(y) == tuple and len(y) == 2 and type(y[0]) == int and type(y[1]) == int: # skip y and fmt
        title  = xlabel
        if type(figsize) == str:
            ylabel = figsize
            xlabel = fmt
        figsize= y
        fmt=None
        y=None
    if callable(y):
        y = np.array([y(i) for i in x])
    if type(fmt) == tuple: # skip fmt
        title  = ylabel
        ylabel = xlabel
        if type(figsize) == str:
            xlabel = figsize
        figsize= fmt
        fmt=None
    elif type(figsize) == str: # skip figsize
        title  = ylabel
        ylabel = xlabel
        xlabel = figsize
        figsize= (10,8)

    if fmt is None:
        fmt = "-"

    if type(x) != np.ndarray:
        x = np.array(list(x))
    if y is not None and type(y) != np.ndarray:
        y = np.array(list(y))
    # plot
    if len(plt.get_fignums()) == 0:
        plt.figure(figsize=figsize)
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    if grid:
        plt.grid()
    if fmt == ".":
        if y is None:
            y = x
            x = np.linspace(1,len(x),len(x))
        if is_complex(y):
            plt.scatter(x, y.real, label="real", **pltargs)
            plt.scatter(x, y.imag, label="imag", **pltargs)
            plt.legend()
        else:
            plt.scatter(x, y, **pltargs)
    elif y is not None:
        if is_complex(y):
            plt.plot(x, y.real, fmt, label="real", **pltargs)
            plt.plot(x, y.imag, fmt, label="imag", **pltargs)
            plt.legend()
        else:
            if len(y.shape) == 1:
                plt.plot(x, y, fmt, **pltargs)
            else:
                for yi in y:
                    plt.plot(x, yi, fmt, **pltargs)
    else:
        if is_complex(x):
            plt.plot(x.real, fmt, label="real", **pltargs)
            plt.plot(x.imag, fmt, label="imag", **pltargs)
            plt.legend()
        else:
            if len(x.shape) == 1:
                plt.plot(x, fmt, **pltargs)
            else:
                for xi in x:
                    plt.plot(xi, fmt, **pltargs)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if "label" in pltargs:
        plt.legend()
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    if save_file:
        plt.savefig(save_file, bbox_inches='tight')

    if show:
        plt.show()

# # basics, no log
# def hist(data, bins=None, xlabel="", title="", density=False):
#     def bins_sqrt(data):
#         return int(np.ceil(np.sqrt(len(data))))
#
#     plt.figure(figsize=(10,5))
#
#     # bins
#     if not bins:
#         bins = bins_sqrt(data)
#     n, bins, _ = plt.hist(data, bins=bins, density=density)
#
#     # visuals
#     plt.title(title)
#     plt.ylabel("Density" if density else "Frequency")
#     plt.xlabel(xlabel)
#     plt.gca().spines["top"].set_visible(False)
#     plt.gca().spines["right"].set_visible(False)
#     plt.gca().spines["bottom"].set_visible(False)
#     return n, bins

def histogram(data, bins=None, xlog=False, density=False):
    """Returns `(n, bins)`, where `n` is the number of data points in each bin and `bins` is the bin edges."""
    if xlog:
        if not hasattr(bins, '__len__'):
            bins = logbins(data, num=bins)
    elif bins is None:
        bins = bins_sqrt(data)
    return np.histogram(data, bins=bins, density=density)

def hist(data, bins=None, xlabel="", title="", xlog=False, ylog=False, density=False, vlines=None, colored=None, cmap="viridis", save_file=None, show=True, figsize=(10,5)):
    """Uses magic to create pretty histograms."""

    if type(bins) == str:
        if bins == "log":
            xlog = True
        elif bins == "loglog":
            xlog = True
            ylog = True
        else:
            xlabel = bins
        bins = None

    # create figure
    if colored:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [10, 1]})
        ax0 = ax[0]
    else:
        fig, ax0 = plt.subplots(figsize=figsize)

    # filter nan, -inf, and inf from data
    data = np.array(data)
    n_filtered = np.sum(np.isnan(data)) + np.sum(np.isinf(data))
    if n_filtered > 0:
        n_original = len(data)
        data = data[~np.isnan(data)] # filter nan
        data = data[~np.isinf(data)] # filter inf and -inf
        print(f"nan or inf values detected in data: {n_filtered} values ({n_filtered/n_original*100:.3f}%) filtered out")

    n, bins = histogram(data, bins=bins, xlog=xlog, density=density)
    ax0.hist(bins[:-1], bins, weights=n) # TODO: moving_avg(bins,2) instead of bins[:-1]?
    if xlog:
        ax0.set_xscale("log")
    if ylog:
        ax0.set_yscale("log")

    # visuals
    ax0.set_title(title)
    ax0.set_ylabel("density" if density else "frequency")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)

    # Add optional vertical lines
    if vlines:
        # if not iterable, put it in a list
        if not hasattr(vlines, '__iter__') or type(vlines) == str:
            vlines = [vlines]
        for v in vlines:
            # if iterable, assume the first to be the number and the rest to be kwargs
            if hasattr(v, '__iter__') and type(v) != str:
                v, kwargs = v[0], v[1:]
            else:
                kwargs = {}
            # if string, interpret it
            if type(v) == str:
                import re
                if v == "mean":
                    v = np.mean(data)
                elif v == "median":
                    v = np.median(data)
                elif v == "mode":
                    # bin with the highest frequency
                    v = bins[np.argmax(n)]
                elif v == "std":
                    v1 = np.mean(data) + np.std(data)
                    v2 = np.mean(data) - np.std(data)
                    if "color" not in kwargs:
                        kwargs["color"] = "black"
                    if "alpha" not in kwargs:
                        kwargs["alpha"] = .2
                    ax0.axvspan(v1, v2, **kwargs)
                    continue
                elif v == "2std":
                    v1 = np.mean(data) + 2*np.std(data)
                    v2 = np.mean(data) - 2*np.std(data)
                    if "color" not in kwargs:
                        kwargs["color"] = "black"
                    if "alpha" not in kwargs:
                        kwargs["alpha"] = .2
                    ax0.axvspan(v1, v2, **kwargs)
                    continue
                # credible interval
                elif re.match(r"\d?\d\.?\d?\d?%?C?I?", v.replace(' ', '').replace('-', '')):
                    v = float(v.replace(' ', '').replace('%', '').replace('-', '').replace('CI', ''))
                    if v < 0 or v > 100:
                        raise ValueError(f"CI must be between 0 and 100, but was {v}")
                    v /= 100
                    v1 = np.quantile(data, (1-v)/2)
                    v2 = np.quantile(data, (1+v)/2)
                    if "color" not in kwargs:
                        kwargs["color"] = "black"
                    if "alpha" not in kwargs:
                        kwargs["alpha"] = .2
                    ax0.axvspan(v1, v2, **kwargs)
                    continue
                else:
                    raise ValueError(f"Unknown vlines string: {v}")
            if "color" not in kwargs:
                kwargs["color"] = "black"
            if "linestyle" not in kwargs:
                kwargs["linestyle"] = "--"
            ax0.axvline(v, **kwargs)
            # add a label at v
            if "label" not in kwargs:
                kwargs["label"] = f"{v:.3f}"
            if kwargs["label"] != None:
                # ax0.text(v + 0.01*(ax0.get_xlim[1] - ax0.get_xlim()[0]), ax0.get_ylim()[1]*0.9, kwargs["label"], rotation=90, verticalalignment="top")
                ax0.text(v, 0, kwargs["label"], rotation=-45, verticalalignment="top")

    # Add colored 1d scatter plot
    if colored:
        ax[1].scatter(data, np.zeros(*data.shape), alpha=.5, c=colored, cmap=cmap, marker="|", s=500)
        # ax[1].axis("off")
        ax[1].set_xlabel(xlabel)
        ax[1].set_yticks([])
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[1].spines["left"].set_visible(False)

        norm = plt.Normalize(vmin=min(colored), vmax=max(colored))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
        cb = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.01, aspect=50)
    else:
        ax0.set_xlabel(xlabel)

    plt.tight_layout()

    if show:
        plt.show()

    if save_file:
        plt.savefig(save_file, bbox_inches='tight')

    return n, bins

def scatter1d(data, figsize=None, xticks=None, alpha=.5, s=500, marker="|", xlim=None, xlabel="", title="", show=True, save_file=None, **pltargs):
    """Create only one axis on which to plot the data."""

    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        figsize = [10,1]
        if xlabel:
            figsize[1] += 0.2
        if title:
            figsize[1] += 0.2
        fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    size = np.prod(np.array(data, copy=False).shape)
    plt.scatter(data, np.zeros(size), alpha=alpha, marker=marker, s=s, **pltargs)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    if xticks:
        ax.set_xticks(xticks)
    if title:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
    if show:
        plt.show()

def colorize_complex(z):
    """Colorize complex numbers by their angle and magnitude."""

    from colorsys import hls_to_rgb

    r = np.abs(z)
    a = np.angle(z)

    h = a / (2*np.pi)
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb)(h,l,s)
    c = np.array(c).transpose(1,2,0) # convert shape (3,n,m) -> (n,m,3)
    return c

def imshow(a, figsize=None, title="", cmap="hot", xticks=None, yticks=None, xticks_rot=0, xlabel=None, ylabel=None, colorbar='auto', magic_reshape=True, show=True, save_file=None, **pltargs):
    """Uses magic to create pretty images from arrays.

    Parameters
        a (np.ndarray):       2D array to be plotted
        figsize (tuple):      Figure size
        title (str):          Title of the plot
        cmap (str):           Colormap to use
        xticks (tuple|list):  List of xticks. If given as tuple, etiher (start, stop) or (ticks, labels). If given as list, ticks are spaced evenly (so, ensure the first and last label are given!). Give an empty list to disable the x-axis.
        yticks (tuple|list):  List of yticks. If given as tuple, etiher (start, stop) or (ticks, labels). If given as list, ticks are spaced evenly (so, ensure the first and last label are given!). Give an empty list to disable the y-axis.
        xticks_rot (float):   Rotation of the xticks
        xlabel (str):         Label for the x-axis
        ylabel (str):         Label for the y-axis
        colorbar (bool|str):  Whether to show the colorbar. If 'auto', show if the array is not complex.
        magic_reshape (bool): If True, automatically reshape long vectors if possible
        show (bool):          Whether to show the plot
        save_file (str):      If given, save the plot to this file
        **pltargs:            Additional arguments to pass to plt.imshow

    Returns
        None
    """

    a = np.array(a, copy=False)
    is_vector = np.prod(a.shape) == np.max(a.shape)

    # magic reshape
    if magic_reshape and is_vector and max(a.shape) >= 100:
        best_divisor = int_sqrt(np.prod(a.shape))
        a = a.reshape(best_divisor, -1)
        is_vector = False
    if len(a.shape) == 1:
        a = a[:,None]   # default to vertical

    # intelligent figsize
    if figsize is None:
        max_dim = 32
        if a.shape[0] >= a.shape[1]:
            xdim, ydim = max_dim, min(max_dim, np.log2(max(2,a.shape[0])))  # matplotlib automatially scales the other dimension
        else:
            xdim, ydim = min(max_dim, np.log2(max(2,a.shape[1]))), max_dim
        figsize = (xdim, ydim)
    fig = plt.figure(figsize=figsize)

    if is_vector:
        if is_complex(a):
            img = colorize_complex(a)
            if colorbar == True:
                print("Warning: colorbar not supported for complex arrays. Use `complex_colorbar()` to see the color reference.")
            plt.imshow(img, aspect=5/a.shape[0], **pltargs)
        else:
            a = a.real
            img = plt.imshow(a, cmap=cmap, **pltargs)
            if colorbar:  # True or 'auto'
                fig.colorbar(img, fraction=0.01, pad=0.01)
    elif len(a.shape) == 2:
        if is_complex(a):
            img = colorize_complex(a)
            if colorbar == True:
                print("Warning: colorbar not supported for complex arrays. Use `complex_colorbar()` to see the color reference.")
            plt.imshow(img, **pltargs)
        else:
            a = a.real
            img = plt.imshow(a, cmap=cmap, **pltargs)
            if colorbar:  # True or 'auto'
                if 1 <= a.shape[1] / a.shape[0] < 2.5:
                    fig.colorbar(img, fraction=0.04 * a.shape[0] / a.shape[1], pad=0.01)
                else:
                    fig.colorbar(img, fraction=0.02, pad=0.01)
    elif len(a.shape) == 3 and a.shape[2] == 3:
        plt.imshow(a, **pltargs)
        if colorbar == True:
            print("Warning: colorbar not supported for RGB images.")
    else:
        raise ValueError(f"Array must be 2D or 1D, but shape was {a.shape}")

    def generate_ticks_and_labels(ticks, shape, max_ticks=10):
        if isinstance(ticks, tuple) and len(ticks) == 2:
            if isinstance(ticks[0], (int, float)):
                ticklabels = np.linspace(ticks[0], ticks[1], min(max_ticks, shape))
                ticklabels = np.round(ticklabels, -int(np.log10(ticks[-1]-ticks[0])-3))
                ticks = np.linspace(0, shape-1, len(ticklabels))
            else:
                ticklabels = ticks[1]
                ticks = ticks[0]
        else:
            if len(ticks) > max_ticks:
                ticklabels = np.linspace(ticks[0], ticks[-1], max_ticks)
                ticklabels = np.round(ticklabels, -int(np.round(np.log10(ticks[-1]-ticks[0])-3)))
            else:
                ticklabels = ticks
            ticks = np.linspace(0, shape-1, len(ticklabels))
        return ticks, ticklabels

    if xticks == False:
        xticks = []
    if xticks is not None:
        xticks, xticklabels = generate_ticks_and_labels(xticks, a.shape[1])
        plt.xticks(xticks, xticklabels, rotation=xticks_rot, ha='center')
    if yticks == False:
        yticks = []
    if yticks is not None:
        yticks, yticklabels = generate_ticks_and_labels(yticks, a.shape[0])
        plt.yticks(yticks, yticklabels)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xticks is not None and len(xticks) == 0 and yticks is not None and len(yticks) == 0:
        plt.axis('off')

    plt.title(title)
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
    if show:
        plt.show()

def complex_colorbar(figsize=(2,2)):
    """Show the color reference plot for complex numbers."""

    imag, real = np.mgrid[-1:1:0.01,-1:1:0.01]
    imag = imag[::-1] # convention: turn counter-clockwise
    x = real + 1j*imag
    c = colorize_complex(x)
    plt.figure(figsize=figsize)
    plt.imshow(c)
    plt.xticks(np.linspace(0,200,5), np.linspace(-1,1,5))
    plt.yticks(np.linspace(0,200,5), np.linspace(-1,1,5)[::-1])
    plt.xlabel("real")
    plt.ylabel("imag")

def bar(heights, log=False, show=True, save_file=None):
    """Uses magic to create pretty bar plots."""
    N = len(heights)
    plt.figure(figsize=(int(np.ceil(N/2)),6))
    plt.bar(range(N), height=heights)
    plt.xticks(range(N))
    if log:
        plt.yscale("log")
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
    if show:
        plt.show()

def rgb(r,g=1.0,b=1.0,a=1.0, as255=False):
    """Converts (r,g,b,a) to a hex string."""
    conv = 1 if as255 else 1/255
    if type(r) == str:
        s = r.split("#")[-1]
        r = int(s[0:2],16)*conv
        g = int(s[2:4],16)*conv
        b = int(s[4:6],16)*conv
        if len(s) > 6:
            a = int(s[6:8],16)*conv
        else:
            a = 1
        return (r,g,b,a)
    return "#" + "".join('{0:02X}'.format(int(v/conv)) for v in [r,g,b,a])

def perceived_brightness(r,g,b,a=None): # a is ignored
    """Returns the perceived brightness of a color given as (r,g,b,a)."""
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

# coloring for pd.DateFrame
# todo: use package "webcolors" (e.g. name to rgb)
def pdcolor(df, threshold=None, minv=None, maxv=None, colors=('#ff0000', '#ffffff', '#069900'), tril_if_symmetric=True, bi=False):
    """Color a pandas DataFrame according to its values. Only works for Jupyter notebooks.
    If `minv` and `maxv` are given, they are used instead of the minimum and maximum values.
    If `tril_if_symmetric` is True, the upper triangle is colored if the matrix is symmetric.
    If `bi` is True, the first color is ignored.

    Args:
        df: The DataFrame to color.
        threshold: If given, the DataFrame is colored based on whether the values are above or below the threshold, using the first and last color of `colors`, respectively.
        minv: The minimum value to use. If None, the minimum value in the DataFrame is used.
        maxv: The maximum value to use. If None, the maximum value in the DataFrame is used.
        colors: A list of colors to use. The first color is used for the lowest values, the last color for the highest values. If `threshold` is given, the first and last color are used for values below and above the threshold, respectively. Else a linear interpolation is used.
        tril_if_symmetric: If True, only the upper triangle is colored if the matrix is symmetric.
        bi: If True, the first color is ignored (i.e., in the default settings, uses white for the lowest values instead of red).

    Returns:
        A styled DataFrame.

    Example: pdcolor(pd.DataFrame(np.random.rand(10,10)))

    """
    def blackorwhite(r,g=None,b=None):
        if g is None:
            r,g,b,a = rgb(r)
        return 'white' if perceived_brightness(r,g,b) < 0.5 else 'black'

    df = df.dropna(thresh=1).T.dropna(thresh=1).T # filter out rows and cols with no data
    if bi:
        colors = colors[1:]
    if threshold:
        def highlight(value):
            if np.isnan(value):
                bg_color = 'white'
                color = 'white'
            elif value < -threshold:
                bg_color = colors[0]
                color = blackorwhite(bg_color)
            elif value > threshold:
                bg_color = colors[-1]
                color = blackorwhite(bg_color)
            else:
                bg_color = 'white'
                color = 'black'
            return f"background-color: %s; color: %s" % (bg_color, color)
    else:
        if len(colors) < 2:
            raise ValueError("Please give at least two colors!")
        if minv is None and maxv is None \
                    and len(df.columns) == len(df.index) and (df.columns == df.index).all() \
                    and df.max().max() <= 1 and df.min().min() >= -1: # corr matrix!
            maxv = 1
            minv = -1
        if not maxv:
            maxv = df.max().max()
        if not minv:
            minv = df.min().min()
        if maxv <= minv:
            raise ValueError(f"Maxv must be higher than minv, but was: %f <= %f" % (maxv, minv))

        def getRGB(v):
            scaled = (v - minv)/(maxv - minv) * (len(colors)-1)
            scaled = max(0,min(scaled, len(colors)-1-1e-10)) #[0;len(colors)-1[
            subarea = int(np.floor(scaled))
            low_c, high_c = colors[subarea], colors[subarea+1] # get frame
            low_c, high_c = np.array(rgb(low_c)), np.array(rgb(high_c)) # convert to (r,b,g,a)
            r,g,b,a = (scaled-subarea)*(high_c-low_c) + low_c
            return rgb(r,g,b,a), blackorwhite(r,g,b)

        def highlight(value):
            if np.isnan(value):
                bg_color = 'white'
                color = 'white'
            else:
                bg_color, color = getRGB(value)
            return f"background-color: %s; color: %s" % (bg_color, color)

    if tril_if_symmetric and is_symmetric(df):
        df = df.where(np.tril(np.ones(df.shape), -1).astype(bool))
        df = df.dropna(thresh=1).T.dropna(thresh=1).T
    return df.style.applymap(highlight)


def graph_from_matrix(A, plot=True, lib="networkx"):
    """Create a graph from a matrix by interpreting the matrix as a weighted adjacency matrix.

    Args:
        A: The matrix to convert.
        plot: If True, plot the graph.
        lib: The library to use. Can be "networkx" (or "nx"), "graphviz" (or "gv"), or "igraph" (or "ig").
    """

    A = np.array(A)
    if A.shape[0] != A.shape[1]:
        # add zeros to make it square
        A = np.pad(A, ((0, max(A.shape)-A.shape[0]), (0, max(A.shape)-A.shape[1])), 'constant')

    sym = is_symmetric(A)

    if lib == "networkx" or lib == "nx":
        import networkx as nx
        if sym:
            G = nx.from_numpy_array(A, create_using=nx.Graph)
        else:
            G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        if plot:
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True)
            labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
            plt.show()
    elif lib == "graphviz" or lib == "gv":
        from graphviz import Graph, Digraph

        if sym:
            G = Graph()
        else:
            G = Digraph()
        for i in range(A.shape[0]):
            G.node(str(i))
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                # if A is symmetric, only add one edge
                if sym and i > j:
                    continue
                if A[i,j] != 0:
                    G.edge(str(i), str(j), label=str(A[i,j]))
        if plot:
            G.view()
    elif lib == "igraph" or lib == "ig":
        import igraph as ig
        G = ig.Graph.Weighted_Adjacency(A, mode=ig.ADJ_UPPER if sym else ig.ADJ_DIRECTED)
        if plot:
            ig.plot(G)
    else:
        raise ValueError(f"Unknown lib: {lib}")

    return G
