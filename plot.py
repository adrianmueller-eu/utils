import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
import numpy as np
from math import log2
from .mathlib import is_complex, is_symmetric, int_sqrt, next_good_int_sqrt
from .data import logbins, bins_sqrt
from .utils import is_iterable

def plot(x, y=None, fmt="-", figsize=(10,8), xlim=(None, None), ylim=(None, None), xlabel="", ylabel="", title="", labels=None, xticks=None, yticks=None, xlog=False, ylog=False, grid=True, vlines=None, hlines=None, show=True, save_file=None, **pltargs):
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
    if xticks is not None:
        if len(xticks) == 2 and len(xticks[0]) == len(xticks[1]):
            plt.xticks(xticks[0], xticks[1])
        else:
            plt.xticks(xticks)
    if yticks is not None:
        if len(yticks) == 2 and len(yticks[0]) == len(yticks[1]):
            plt.yticks(yticks[0], yticks[1])
        else:
            plt.yticks(yticks)
    if grid and (show or save_file is not None):
        plt.grid()
    if fmt == ".":
        if y is None:
            y = x
            x = np.linspace(1,len(x),len(x))
        if is_complex(y):
            if labels is not None:
                warnings.warn("labels are not supported for complex data", stacklevel=2)
            plt.scatter(x, y.real, label="real", **pltargs)
            plt.scatter(x, y.imag, label="imag", **pltargs)
            plt.legend()
        else:
            if len(y.shape) == 1:
                if labels is not None:
                    if type(labels) == str or not is_iterable(labels):
                        pltargs["label"] = labels
                    else:  # is_iterable(labels):
                        pltargs["label"] = labels[0]
                plt.scatter(x, y, marker=fmt, **pltargs)
            else:
                if labels is not None:
                    assert len(labels) == len(y), f"Number of labels ({len(labels)}) must match number of data vectors ({len(y)})"
                    assert "label" not in pltargs, "label argument is not supported when labels is given"
                    for yi, label in zip(y, labels):
                        plt.scatter(x, yi, marker=fmt, label=label, **pltargs)
                else:
                    for yi in y:
                        plt.scatter(x, yi, marker=fmt, **pltargs)
    elif y is not None:
        if is_complex(y):
            if labels is not None:
                warnings.warn("labels are not supported for complex data", stacklevel=2)
            plt.plot(x, y.real, fmt, label="real", **pltargs)
            plt.plot(x, y.imag, fmt, label="imag", **pltargs)
            plt.legend()
        else:
            if len(y.shape) == 1:
                if labels is not None:
                    if type(labels) == str or not is_iterable(labels):
                        pltargs["label"] = str(labels)
                    else:  # is_iterable(labels):
                        pltargs["label"] = str(labels[0])
                plt.plot(x, y, fmt, **pltargs)
            else:
                if labels is not None:
                    assert len(labels) == len(y), f"Number of labels ({len(labels)}) must match number of data vectors ({len(y)})"
                    assert "label" not in pltargs, "label argument is not supported when labels is given"
                    for yi, label in zip(y, labels):
                        plt.plot(x, yi, fmt, label=label, **pltargs)
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
    if "label" in pltargs or labels is not None:
        plt.legend()
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    if vlines:
        if not hasattr(vlines, '__iter__') or type(vlines) == str:
            vlines = [vlines]
        for v in vlines:
            # if iterable, assume the first to be the number and the rest to be kwargs for axvline
            if hasattr(v, '__iter__') and type(v) != str:
                v, kwargs = v[0], v[1:]
            else:
                kwargs = {}
            if "color" not in kwargs:
                kwargs["color"] = "red"
            if "linestyle" not in kwargs:
                kwargs["linestyle"] = "--"
            plt.axvline(v, **kwargs)
            # add a label at v
            if "label" not in kwargs:
                kwargs["label"] = f"{v:.3f}"
            if kwargs["label"] != None:
                # ax0.text(v + 0.01*(ax0.get_xlim[1] - ax0.get_xlim()[0]), ax0.get_ylim()[1]*0.9, kwargs["label"], rotation=90, verticalalignment="top")
                plt.text(v, 0, kwargs["label"], rotation=-45, verticalalignment="top")
    if hlines:
        if not hasattr(hlines, '__iter__') or type(hlines) == str:
            hlines = [hlines]
        for i, h in enumerate(hlines):
            # if iterable, assume the first to be the number and the rest to be kwargs for axhline
            if hasattr(h, '__iter__') and type(h) != str:
                h, kwargs = h[0], h[1:]
            else:
                kwargs = {}
            if "color" not in kwargs:
                kwargs["color"] = plt.cm.tab10(i)
            if "linestyle" not in kwargs:
                kwargs["linestyle"] = "--"
            plt.axhline(h, **kwargs)
            # add a label at h
            if "label" not in kwargs:
                kwargs["label"] = f"{h:.3f}"
            if kwargs["label"] != None:
                plt.text(0, h, kwargs["label"], verticalalignment="top")

    if save_file:
        plt.savefig(save_file, bbox_inches='tight')

    if show:
        plt.show()

# # basics, no log
# def hist(data, bins=None, xlabel="", title="", density=False):
#     def bins_sqrt(data):
#         return int(np.ceil(sqrt(len(data))))
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

def hist(data, bins=None, xlabel="", title="", labels=None, xlog=False, ylog=False, density=False, vlines=None, colored=None, cmap="viridis", save_file=None, show=True, figsize=(10,5)):
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
    elif len(plt.get_fignums()) == 0:
        fig, ax0 = plt.subplots(figsize=figsize)
    else:
        ax0 = plt.gca()

    def clean_data(data):
        # filter nan, -inf, and inf from data
        data = np.asarray(data)
        nan_filter = np.isnan(data) | np.isinf(data)
        n_filtered = np.sum(nan_filter)
        if n_filtered > 0:
            n_original = len(data)
            data = data[~nan_filter]  # filter out nan and inf
            print(f"nan or inf values detected in data: {n_filtered} values ({n_filtered/n_original*100:.3f}%) filtered out")
        if xlog:
            filter0 = data <= 0
            n_filtered = np.sum(filter0)
            if n_filtered > 0:
                n_original = len(data)
                data = data[~filter0]
                print(f"xlog active, but non-positive values detected in data: {n_filtered} values ({n_filtered/n_original*100:.3f}%) filtered out")
        return data

    data = clean_data(data)
    n, bins = histogram(data.ravel(), bins=bins, xlog=xlog, density=density)
    if len(data.shape) > 1 and 1 < data.shape[0] < 10: # not more than 10 distributions
        if labels is None:
            labels = [None] * data.shape[0]
        for d, label in zip(data, labels):
            ax0.hist(d.ravel(), bins=bins, density=density, alpha=1.5/data.shape[0], label=label)
        ax0.legend()
    else:
        ax0.hist(bins[:-1], bins, weights=n, density=density, label=labels)
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
    size = np.prod(np.asarray(data).shape)
    plt.scatter(data, np.zeros(size), alpha=alpha, marker=marker, s=s, **pltargs)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    if xticks:
        if len(xticks) == 2 and len(xticks[0]) == len(xticks[1]):
            ax.set_xticks(xticks[0], xticks[1])
        else:
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

def scatter(a, b=None, figsize=(6,6), hist_ratio=0.2, x_label="", y_label="", title="", labels=None, save_fig=None, **scatter_kwargs):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize,
                                    gridspec_kw={'width_ratios': [1, hist_ratio], 'height_ratios': [hist_ratio, 1]})

    fig.subplots_adjust(wspace=0, hspace=0)  # remove space between subplots
    fig.suptitle(title)
    fig.subplots_adjust(top=0.95)  # reduce space to suptitle
    ax2.axis('off')  # hide right upper subplot completely

    # scatter plot
    if is_iterable(a) and not is_iterable(a[0]):
        a = [a]
    if b is None:
        if is_complex(a[0]):
            b = [np.imag(a) for a in a]
            a = [np.real(a) for a in a]
            x_label = x_label or "Re"
            y_label = y_label or "Im"
        else:
            raise ValueError("Scatter plot requires two real 1d arrays")
    if labels is None:
        labels = [None]*len(a)
    for ai, bi, label in zip(a, b, labels):
        ax3.scatter(ai, bi, label=label, **scatter_kwargs)
    if labels[0] is not None:
        ax3.legend()
    ax3.set_xlabel(x_label)
    ax3.set_ylabel(y_label)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_color('lightgrey')
    ax3.spines['top'].set_linewidth(0.5)

    # histogram on the top
    a_bins = max([bins_sqrt(ai) for ai in a])
    a_con = np.concatenate(a)
    n, a_bins = histogram(a_con.ravel(), bins=a_bins)
    for ai in a:
        ax1.hist(ai, bins=a_bins, alpha=0.6 if len(a) > 1 else 1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_xticks([])

    # histogram on the right
    b_bins = max([bins_sqrt(bi) for bi in b])
    b_con = np.concatenate(b)
    n, b_bins = histogram(b_con.ravel(), bins=b_bins)
    for bi in b:
        ax4.hist(bi, bins=b_bins, orientation='horizontal', alpha=0.6 if len(a) > 1 else 1, align='mid')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_color('lightgrey')
    ax4.spines['left'].set_linewidth(0.5)
    ax4.set_yticks([])
    ax4.tick_params(axis='x', rotation=45)
    if save_fig is not None:
        fig.savefig(save_fig, bbox_inches='tight')
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
    c = np.asarray(c).transpose(1,2,0) # convert shape (3,n,m) -> (n,m,3)
    return c

def auto_figsize(x_shape, y_shape, max_dim=16):
    if x_shape >= y_shape:
        ydim = min(max_dim, log2(max(2,x_shape)))
        # matplotlib automatially scales the other dimension in the figure
        # but unfortunately not the window when using ipython, so we have to scale it manually
        xdim = ydim * y_shape / x_shape
    else:
        xdim = min(max_dim, log2(max(2,y_shape)))
        ydim = xdim * x_shape / y_shape
    return (xdim, ydim)

def imshow_clean(a, figsize=None, title="", cmap='hot', xlabel=None, ylabel=None, colorbar=False, vmin=None, vmax=None, magic_reshape=True, show=True, save_file=None, **pltargs):
    return imshow(a, xticks=[], yticks=[], figsize=figsize, title=title, cmap=cmap, xlabel=xlabel, ylabel=ylabel, colorbar=colorbar, vmin=vmin, vmax=vmax, magic_reshape=magic_reshape, show=show, save_file=save_file, **pltargs)

def imshow(a, figsize=None, title="", cmap='hot', xticks=None, yticks=None, xticks_rot=0, xlabel=None, ylabel=None, colorbar='auto', vmin=None, vmax=None, magic_reshape=True, show=True, save_file=None, **pltargs):
    """Uses magic to create pretty images from arrays.

    Parameters
        a (np.ndarray):       2D array to be plotted
        figsize (tuple):      Figure size
        title (str):          Title of the plot
        cmap (str):           Colormap to use. For non-complex arrays only.
        xticks (tuple|list):  List of xticks. If given as tuple, either (start, stop) or (ticks, labels). If given as list, ticks are spaced evenly (so, ensure the first and last label are given!). Give an empty list to disable the x-axis.
        yticks (tuple|list):  List of yticks. If given as tuple, either (start, stop) or (ticks, labels). If given as list, ticks are spaced evenly (so, ensure the first and last label are given!). Give an empty list to disable the y-axis.
        xticks_rot (float):   Rotation of the xticks
        xlabel (str):         Label for the x-axis
        ylabel (str):         Label for the y-axis
        vmin / vmax (float):  Minimum and maximum value for the colorbar.
        colorbar (bool|str):  Whether to show the colorbar. If 'auto', show if the array is not complex.
        magic_reshape (bool): If True, automatically reshape long vectors if possible
        show (bool):          Whether to show the plot
        save_file (str):      If given, save the plot to this file
        **pltargs:            Additional arguments to pass to plt.imshow

    Returns
        None
    """

    a = np.asarray(a)
    n_samples = np.prod(a.shape)
    is_vector = n_samples == np.max(a.shape)

    # magic reshape
    if magic_reshape and is_vector and max(a.shape) >= 100:
        m = next_good_int_sqrt(n_samples, p=0.1)
        best_divisor = int_sqrt(m)
        if m != n_samples:
            if np.issubdtype(a.dtype, np.integer):
                fill_val = a.min()
            else:
                fill_val = np.nan
            warnings.warn(f"No good divisor found for magic reshaping {a.shape}. Filling with {m-n_samples}*{[fill_val]} to {m//best_divisor}x{best_divisor}.", stacklevel=2)
            a = np.pad(a, (0, m-n_samples), 'constant', constant_values=fill_val)
        # best_divisor = int_sqrt(n_samples)
        # if best_divisor**2 < n_samples*0.1:
        #     warnings.warn(f"No good divisor found for reshaping {a.shape} to a square. Using the best one: {best_divisor}x{n_samples//best_divisor}", stacklevel=2)
        a = a.reshape(best_divisor, -1)
        is_vector = False
    if len(a.shape) == 1:
        a = a[:,None]   # default to vertical

    # intelligent figsize
    if figsize is None:
        figsize = auto_figsize(*a.shape)
    fig = plt.figure(figsize=figsize)

    iscomplex = is_complex(a)
    if iscomplex:
        if colorbar == True:
            warnings.warn("colorbar not supported for complex arrays. Use `complex_colorbar()` to see the color reference.", stacklevel=2)
        if cmap != 'hot':
            warnings.warn("Argument cmap is not used for complex arrays.", stacklevel=2)

    if vmin is not None or vmax is not None:
        if iscomplex:
            warnings.warn("vmin and vmax are not supported for complex arrays.", stacklevel=2)
            norm = None
        else:
            if vmin is None:
                vmin = np.min(a)
            if vmax is None:
                vmax = np.max(a)
            if vmin > vmax:
                warnings.warn("vmin > vmax", stacklevel=2)
            norm = colors.Normalize(vmin, vmax)
    else:
        norm = None

    if is_vector:
        if iscomplex:
            img = colorize_complex(a)
            plt.imshow(img, aspect=5/a.shape[0], **pltargs)
        else:
            a = a.real
            img = plt.imshow(a, cmap=cmap, norm=norm, **pltargs)
            if colorbar:  # True or 'auto'
                fig.colorbar(img, fraction=0.01, pad=0.01)
    elif len(a.shape) == 2:
        if iscomplex:
            img = colorize_complex(a)
            plt.imshow(img, **pltargs)
        else:
            a = a.real
            img = plt.imshow(a, cmap=cmap, norm=norm, **pltargs)
            if colorbar:  # True or 'auto'
                if 1 <= a.shape[1] / a.shape[0] < 2.5:
                    fig.colorbar(img, fraction=0.04 * a.shape[0] / a.shape[1], pad=0.01)
                else:
                    fig.colorbar(img, fraction=0.02, pad=0.01)
    elif len(a.shape) == 3 and a.shape[2] == 3:
        plt.imshow(a, **pltargs)
        if colorbar == True:
            warnings.warn("colorbar not supported for RGB images.", stacklevel=2)
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

def plot_dynamic(gen, figsize=None, sleep=0, cb=None, skip=0, xlim=None, ylim=None, linewidth=1, dot=None):
    """ Plot a dynamic function.

    Args:
        gen: The generator that yields the next function values.
        figsize: The size of the figure.
        sleep: The time to sleep between frames.
        cb: A callback function that is called for each frame. It can return the modified function values.
        skip: The number of iterations to skip between two frames.

    Returns:
        The animation object.
    """
    # close all previous figures
    plt.close('all')

    if figsize is None:
        figsize = (6,4)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    xs, ys = [], []
    line, = ax.plot([], [], linewidth=linewidth)
    if dot is not None:
        dot, = ax.plot([], [], dot, markersize=3)

    t = 0
    def animate(x):
        nonlocal xs, ys, t
        x, y = x
        if skip > 0:
            for _ in range(skip):
                xs.append(x)
                ys.append(y)
                x, y = next(gen)
                t += 1
        xs.append(x)
        ys.append(y)
        if cb is not None:
            r = cb(t, xs, ys, ax, fig)
            if r is not None:
                xs, ys = r
        t += 1
        line.set_data(xs, ys)
        if dot is not None:
            dot.set_data([x], [y])
        return line,

    return FuncAnimation(fig, animate, frames=gen, interval=sleep, blit=True, repeat=False, cache_frame_data=False)

def imshow_dynamic(gen, figsize=None, sleep=0, cb=None, skip=0):
    """Plot a dynamic image sequence.

    Args:
        gen: The generator that yields the images.
        figsize: The size of the figure.
        sleep: The time to sleep between frames.
        cb: A callback function that is called for each frame. It can return the modified image.
        skip: The number of iterations to skip between two frames.

    Returns:
        The animation object.
    """
    # close all previous figures
    plt.close('all')
    first = next(gen)

    if figsize is None:
        figsize = auto_figsize(*np.asarray(first).shape)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    fig.tight_layout()

    im = ax.imshow(first, cmap='hot')
    ax.axis('off')
    plt.show()

    t = 0
    if cb is not None:
        x = cb(t, first, im, fig)
        t += 1
        if x is not None:
            im.set_data(x)

    def animate(x):
        nonlocal t
        if skip > 0:
            for _ in range(skip):
                x = next(gen)
                t += 1
        if cb is not None:
            r = cb(t, x, im, fig)
            if r is not None:
                x = r
        t += 1
        im.set_data(x)
        return im,

    return FuncAnimation(fig, animate, frames=gen, interval=sleep, blit=True, repeat=False, cache_frame_data=False)

def complex_colorbar(figsize=(2,2), colorizer=colorize_complex):
    """Show the color reference plot for complex numbers."""
    if not isinstance(figsize, (tuple, list)):
        figsize = (figsize, figsize)
    assert isinstance(figsize[0], (int, float)), "figsize must be a number or a tuple"
    assert callable(colorizer), "colorizer must be a callable function"

    imag, real = np.mgrid[-1:1:0.01,-1:1:0.01]
    imag = imag[::-1] # convention: turn counter-clockwise
    x = real + 1j*imag
    c = colorizer(x)
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
            low_c, high_c = np.asarray(rgb(low_c)), np.asarray(rgb(high_c)) # convert to (r,b,g,a)
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

    A = np.asarray(A)
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
