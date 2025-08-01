import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import NullFormatter

import numpy as np
from math import log2, log10, ceil, prod, floor
from .mathlib import is_complex, is_symmetric, int_sqrt, next_good_int_sqrt
from .data import logbins, bins_sqrt
from .utils import is_iterable, as_list_not_str, warn, is_numeric

def poly_scale(y, deg=1):
    if deg == 1:
        return y
    if isinstance(y, (float, int)):
        if y < 0:
            return -(-y)**deg
        return y**deg
    if isinstance(y, complex):
        return y**deg

    def _poly_scale(y):
        # assert isinstance(y, np.ndarray)
        if is_complex(y):
            return y**deg
        y_neg = y < 0
        y_neg_res = -(-y[y_neg])**deg
        y_ = np.zeros_like(y, dtype=y_neg_res.dtype)  # infer float dtype if necessary
        y_[y_neg] = y_neg_res
        y_[~y_neg] = y[~y_neg]**deg
        return y_

    if isinstance(y, np.ndarray):
        return _poly_scale(y)
    elif is_numeric(y[0]):  # already know y is not numeric
        return _poly_scale(np.asarray(y))
    return [poly_scale(yi) for yi in y]

def plot(x, y=None, fmt="-", figsize=(10,8), xlim=(None, None), ylim=(None, None),  xlog=False, ylog=False, grid=True, ypoly=1,
         xlabel="", ylabel="", title="", labels=None, xticks=None, yticks=None, ste=True, capsize=5, show_data_points=True,
         vlines=None, hlines=None, area_quantiles=0.99, area_alpha=0.2, cloud_alpha=0.8, cloud_s=3, max_data_sets=12,
         show=True, save_file=None, **pltargs):
    """Uses magic to create pretty plots."""
    # arg parsing
    if labels is not None:
        labels = as_list_not_str(labels)
    if area_quantiles is None or area_quantiles == 0:
        area_quantiles = None
    elif isinstance(area_quantiles, (float, int)):
        assert 0 < area_quantiles <= 1, f"Invalid percentage: {area_quantiles}"
        lower = (1 - area_quantiles)/2
        upper = 1 - lower
        area_quantiles = (lower,upper)

    if y is None:
        y, x = x, None

    # data preparation
    assert len(y) > 0, "Empty data array"
    scatter_mode = not is_numeric(y[0]) and not is_numeric(y[0][0])
    to_test_for_complex = y[0][0] if scatter_mode else y[0]
    if is_complex(to_test_for_complex):
        if is_numeric(y[0]):
            y = [np.asarray(y)]
        else:
            y = [np.asarray(yi) for yi in y]  # assume scatter dimension all equal length for complex data (for now)
        assert y[0].ndim in (0,1), "Complex data must be 1D or 2D"
        assert labels is None, "labels are not supported for complex data"
        y = [np.asarray(r_or_im) for r_or_im in zipl((yi.real, yi.imag) for yi in y)]
        labels = ["Re", "Im"]
    elif (scatter_mode or (not is_numeric(y[0]) and is_numeric(y[0][0]) and len(y) <= max_data_sets)) and \
        (x is None or is_numeric(x[0]) and not is_numeric(y[0]) and len(x) >= len(y[0]) or not is_numeric(x[0]) and len(x) == len(y)):
        # print(len(y), "datasets")
        # y is a list of data sets to plot individually
        if x is None:
            x = [np.arange(len(yi)) for yi in y]
        elif is_numeric(x[0]):
            x = [x[:len(yi)] for yi in y]
    else:
        # print("One dataset")
        # single dataset (single color)
        if x is None:
            x = np.arange(len(y))
        x = [x[:len(y)]]
        y = [y]

    for i in range(len(y)):
        assert len(x[i]) == len(y[i]), f"x and y in dataset {i} do not have the same length: {len(x[i])} < {len(y[i])}"
        assert is_numeric(x[i][0]), f"x of dataset {i} must be 1D"
    scatter_mode = not is_numeric(y[0]) and not is_numeric(y[0][0])

    if "label" in pltargs:
        assert labels is None, "label argument is not supported when labels is given"
        labels = [pltargs["label"]]
    elif labels is None:
        labels = [None]*len(y)
    assert len(labels) == len(y), f"Number of labels ({len(labels)}) must match number of data vectors ({len(y)})"

    if fmt != '.':
        if 's' in pltargs:
            cloud_s = pltargs.pop('s')
        y_stes = None
        if scatter_mode:
            if isinstance(ste, bool) and ste:
                y_stes = [[np.std(yij)/np.sqrt(len(yij)) for yij in yi] for yi in y]

            if area_quantiles is not None:
                area_quantiles_ = []
                for yi in y:
                    lower = [np.quantile(yij, area_quantiles[0]) for yij in yi]
                    upper = [np.quantile(yij, area_quantiles[1]) for yij in yi]
                    area_quantiles_.append((lower, upper))
                area_quantiles = area_quantiles_
        if not is_numeric(ste):
            if not is_numeric(ste[0]):
                y_stes = [np.asarray(ste_i) for ste_i in ste]
            else:
                y_stes = [np.asarray(ste)]
            for i, (yi, ystei) in enumerate(zip(y, y_stes)):
                assert len(ystei) == len(yi), f"Number of ste_{i} ({len(ystei)}) must match y_{i} ({len(yi)})"
        if y_stes is not None and len(x[0]) == 1 and fmt == "-":
            fmt = "x"  # show the mean in the error bar plot

    y = poly_scale(y, 1/ypoly)

    # plot
    assert len(y) <= max_data_sets, f"Please don't plot more than {max_data_sets} sets of data points simultaneously."
    if len(plt.get_fignums()) == 0:
        plt.figure(figsize=figsize)
    if fmt == ".":
        for xi, yi, label in zip(x, y, labels):
            plt.scatter(xi, yi, marker=fmt, label=label, **pltargs)
    else:
        for i, (xi, yi) in enumerate(zip(x,y)):
            if scatter_mode:
                if area_quantiles is not None:
                    lower, upper = area_quantiles[i]
                    plt.fill_between(xi, lower, upper, alpha=area_alpha, color=plt.cm.tab10(i))
                if show_data_points:
                    for j, yij in enumerate(yi):
                        plt.scatter([xi[j]]*len(yij), yij, alpha=cloud_alpha, s=cloud_s, color=plt.cm.tab10(i))
                yi = [np.mean(yij) for yij in yi]
            if y_stes is None:
                plt.plot(xi, yi, fmt, label=labels[i], **pltargs)
            else:
                y_2stes_i = 2*np.array(y_stes[i])
                plt.errorbar(xi, yi, yerr=y_2stes_i, fmt=fmt, label=labels[i], capsize=capsize, **pltargs)

    if isinstance(xlim, tuple) and (xlim[0] is not None or xlim[1] is not None):
        plt.xlim(xlim)
    if isinstance(ylim, tuple) and (ylim[0] is not None or ylim[1] is not None):
        plt.ylim(ylim)
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
    if grid:
        plt.grid()

    if ypoly != 1:
        ax = plt.gca()
        yticks = ax.get_yticks()
        yticks_scaled = poly_scale(yticks, ypoly)

        # if all powers of 10, write them as 10^x
        if (all(yticks_scaled > 0) or all(yticks_scaled < 0)) and np.allclose(np.log10(np.abs(yticks_scaled)) % 1, 0):
            sign = '-' if (yticks_scaled[0] < 0) else ''
            formatter = lambda y, pos=None: f"${sign}10^{{{ypoly*int(log10(y))}}}$"
        else:
            formatter = lambda y, pos=None: f"${poly_scale(y, ypoly):.6g}$"
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_formatter(NullFormatter())

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if "label" in pltargs or labels is not None and not all(l is None for l in labels):
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
#         return ceil(sqrt(len(data))
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

def clean_hist_data(data, log=False, lim=None):
    # filter nan, -inf, and inf from data
    data = np.asarray(data)
    nan_filter = np.isnan(data) | np.isinf(data)
    n_filtered = np.sum(nan_filter)
    if n_filtered > 0:
        n_original = len(data)
        data = data[~nan_filter]
        warn(f"nan or inf values detected in data: {n_filtered} values ({n_filtered/n_original:.3%}) filtered out")
    # filter out invalid data for log scale
    if log:
        filter0 = data <= 0
        n_filtered = np.sum(filter0)
        if n_filtered > 0:
            n_original = len(data)
            data = data[~filter0]
            warn(f"log active, but non-positive values detected in data: {n_filtered} values ({n_filtered/n_original:.3%}) filtered out")
    # filter out data outside of lim
    if lim is not None:
        xmin, xmax = np.min(data), np.max(data)
        xmin_, xmax_ = lim
        if xmin_ is not None:
            xmin = max(xmin, xmin_)
        if xmax_ is not None:
            xmax = min(xmax, xmax_)
        # n_before = sum([len(d) for d in data])
        data = [di[(xmin <= di) & (di <= xmax)] for di in data]
        # n_total = sum([len(d) for d in data])
        # if n_total < 0.99*n_before:
        #     print(f"Filtered {n_before - n_total} points from data ({(n_before - n_total)/n_before:.2%})")
    return data

def clean_data_2d(x, y, xlog=False, ylog=False, xlim=None, ylim=None):
    # filter nan, -inf, and inf from data
    x = np.asarray(x)
    y = np.asarray(y)
    nan_filter = np.isnan(x) | np.isinf(x) | np.isnan(y) | np.isinf(y)
    n_filtered = np.sum(nan_filter)
    if n_filtered > 0:
        n_original = len(x)
        x = x[~nan_filter]
        y = y[~nan_filter]
        warn(f"nan or inf values detected in data: {n_filtered} values ({n_filtered/n_original:.3%}) filtered out")
    # filter out invalid data for log scale
    if xlog:
        filter0 = x <= 0
        n_filtered = np.sum(filter0)
        if n_filtered > 0:
            n_original = len(x)
            x = x[~filter0]
            y = y[~filter0]
            warn(f"xlog active, but non-positive values detected in data: {n_filtered} values ({n_filtered/n_original:.3%}) filtered out")
    if ylog:
        filter0 = y <= 0
        n_filtered = np.sum(filter0)
        if n_filtered > 0:
            n_original = len(y)
            x = x[~filter0]
            y = y[~filter0]
            warn(f"ylog active, but non-positive values detected in data: {n_filtered} values ({n_filtered/n_original:.3%}) filtered out")
    # filter out data outside of xlim
    if xlim is not None:
        xmin, xmax = np.min(x), np.max(x)
        xmin_, xmax_ = xlim
        if xmin_ is not None:
            xmin = max(xmin, xmin_)
        if xmax_ is not None:
            xmax = min(xmax, xmax_)
        mask = (xmin <= x) & (x <= xmax)
        x = x[mask]
        y = y[mask]
    # filter out data outside of ylim
    if ylim is not None:
        ymin, ymax = np.min(y), np.max(y)
        ymin_, ymax_ = ylim
        if ymin_ is not None:
            ymin = max(ymin, ymin_)
        if ymax_ is not None:
            ymax = min(ymax, ymax_)
        mask = (ymin <= y) & (y <= ymax)
        x = x[mask]
        y = y[mask]
    return x, y

def hist(data, bins=None, xlabel="", title="", labels=None, xlog=False, ylog=False, density=False, xlim=None, vlines=None, colored=None, cmap="viridis", save_file=None, show=True, figsize=(10,5)):
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

    data = clean_hist_data(data, log=xlog, lim=xlim)
    n, bins = histogram(data.ravel(), bins=bins, xlog=xlog, density=density)
    if len(data.shape) > 1 and 1 < data.shape[0] < 10: # not more than 10 distributions
        if labels is None:
            labels = [None] * data.shape[0]
        for d, label in zip(data, labels):
            ax0.hist(d.ravel(), bins=bins, density=density, alpha=1.5/data.shape[0], label=label)
        if not all(l is None for l in labels):
            ax0.legend()
    else:
        ax0.hist(bins[:-1], bins, weights=n, density=density, label=labels)
    if xlog:
        ax0.set_xscale("log")
    if ylog:
        ax0.set_yscale("log")
    if xlim is not None:
        ax0.set_xlim(xlim)

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

def _simple_hist(ax, data, xlog=False, **hist_kwargs):
    a_bins = max([bins_sqrt(ai) for ai in data])
    a_con = np.concatenate(data)
    n, a_bins = histogram(a_con.ravel(), bins=a_bins, xlog=xlog)
    for ai in data:
        ax.hist(ai, bins=a_bins, alpha=0.6 if len(data) > 1 else 1, **hist_kwargs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if xlog:
        if 'orientation' in hist_kwargs and hist_kwargs['orientation'] == 'horizontal':
            ax.set_yscale('log')
        else:
            ax.set_xscale('log')

def scatter1d(data, figwidth=6, xlabel="", title="", hist='auto', xlim=None, xlog=False, xticks=None, alpha=None, s=500, marker="|", save_file=None, show=True, **scatter_kwargs):
    """Create only one axis on which to plot the data."""

    # prepare data
    if is_iterable(data) and not is_iterable(data[0]):  # arrays may have different lengths -> no numpy array!
        data = [data]
    assert len(data) <= 12, "Please don't plot more than 12 sets of data points simultaneously."
    data = [np.asarray(d) for d in data]  # may have different lengths
    for d in data:
        assert d.ndim == 1, f"Data must be 1D, but was {d.shape}"

    data = clean_hist_data(data, log=xlog, lim=xlim)
    if hist == 'auto':
        hist = sum([len(d) for d in data]) >= 1000

    # create figure
    figsize = [figwidth,1]
    if hist:
        figsize[1] *= 2
    if xlabel:
        figsize[1] += 0.2
    if title:
        figsize[1] += 0.2
    if hist:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 1]})
    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

    # plotting
    if alpha is None:
        alpha = 500/len(data)/len(data[0])
        alpha = min(1, max(0.002, alpha))
    for xi in data:
        ax.scatter(xi, np.zeros(len(xi)), alpha=alpha, marker=marker, s=s, **scatter_kwargs)

    # other arguments
    if xticks:
        if len(xticks) == 2 and len(xticks[0]) == len(xticks[1]):
            ax.set_xticks(xticks[0], xticks[1])
        else:
            ax.set_xticks(xticks)
    if xlog:
        ax.set_xscale('log')
    if xlim is not None:
        ax.set_xlim(xlim)
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)

    if hist:
        # histogram on the bottom
        _simple_hist(ax2, data, xlog=xlog)
        ax2.set_xticks([])
        if xlim is not None:
            ax2.set_xlim(xlim)

    # visuals
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()

    # finalize
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
    if show:
        plt.show()

def scatter(a, b=None, figsize=(6,6), xlabel="", ylabel="", title="", hist='auto', hist_ratio=0.2,
            xlim=None, ylim=None, xlog=False, ylog=False, xticks=None, yticks=None, labels=None,
            alpha=None, s=3, marker='.', save_fig=None, show=True, **scatter_kwargs):
    # prepare data
    if is_iterable(a) and not is_iterable(a[0]):  # arrays may have different lengths -> no numpy array!
        a = [a]
    if b is None:
        if is_complex(a[0]):
            b = [np.imag(a) for a in a]
            a = [np.real(a) for a in a]
            xlabel = xlabel or "Re"
            ylabel = ylabel or "Im"
        elif hasattr(a[0], '__len__') and len(a[0]) == 2:
            if hasattr(a, 'shape'):
                a, b = a[None,:,0], a[None,:,1]
            else:
                a, b = zip(*a)
                a, b = [a], [b]
        elif hasattr(a[0], '__len__') and hasattr(a[0][0], '__len__'):
            assert len(a[0][0]) == 2, f"Can only plot 2D data, but got {len(a[0][0])}"
            a_, b_ = [], []
            for ai in a:
                a, b = zip(*ai)
                a_.append(a)
                b_.append(b)
            a, b = a_, b_
        else:
            assert not ylabel, "ylabel is not supported for 1D data"
            assert ylim is None, "ylim is not supported for 1D data"
            assert yticks is None, "yticks is not supported for 1D data"
            assert labels is None, "labels is not supported for 1D data"
            # use scatter1d default values
            if "figwidth" in scatter_kwargs:
                figwidth = scatter_kwargs["figwidth"]
                del scatter_kwargs["figwidth"]
            else:
                figwidth = figsize[0]
            if s == 3:
                s = 500
            if marker == '.':
                marker = '|'
            return scatter1d(a, figwidth=figwidth, xlabel=xlabel, title=title, hist=hist, xlim=xlim, xlog=xlog, xticks=xticks,
                             alpha=alpha, s=s, marker=marker, save_file=save_fig, show=show, **scatter_kwargs)
    assert len(a) <= 12, "Please don't plot more than 12 sets of data points simultaneously."
    for i, (ai, bi) in enumerate(zip(a, b)):
        a[i], b[i] = clean_data_2d(ai, bi, xlog=xlog, ylog=ylog, xlim=xlim, ylim=ylim)
    if hist == 'auto':
        hist = True  # always show by default

    # create figure
    if hist:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize,
                                        gridspec_kw={'width_ratios': [1, hist_ratio], 'height_ratios': [hist_ratio, 1]})

        fig.subplots_adjust(wspace=0, hspace=0)  # remove space between subplots
        fig.suptitle(title)
        fig.subplots_adjust(top=0.95)  # reduce space to suptitle
        ax2.axis('off')  # hide right upper subplot completely
    else:
        plt.figure(figsize=figsize)
        ax3 = plt.gca()

    # plotting
    if alpha is None:
        alpha = 10000/sum(len(ai) for ai in a)
        alpha = min(1, max(0.002, alpha))
    if labels is None:
        labels = [None]*len(a)
    for ai, bi, label in zip(a, b, labels):
        ax3.scatter(ai, bi, label=label, alpha=alpha, marker=marker, s=s, **scatter_kwargs)

    # other arguments
    if "label" in scatter_kwargs or not all(l is None for l in labels):
        ax3.legend()
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(ylabel)
    if xticks:
        if len(xticks) == 2 and len(xticks[0]) == len(xticks[1]):
            ax3.set_xticks(xticks[0], xticks[1])
        else:
            ax3.set_xticks(xticks)
    if yticks:
        if len(yticks) == 2 and len(yticks[0]) == len(yticks[1]):
            ax3.set_yticks(yticks[0], yticks[1])
        else:
            ax3.set_yticks(yticks)
    if xlog:
        ax3.set_xscale('log')
    if ylog:
        ax3.set_yscale('log')
    if xlim is not None:
        ax3.set_xlim(xlim)
    if ylim is not None:
        ax3.set_ylim(ylim)

    if hist:
        # histogram on the top
        _simple_hist(ax1, a, xlog=xlog)
        ax1.set_xticks([])
        if xlim is not None:
            ax1.set_xlim(xlim)

        # histogram on the right
        _simple_hist(ax4, b, xlog=ylog, orientation='horizontal', align='mid')
        ax4.spines['left'].set_visible(True)
        ax4.spines['left'].set_color('lightgrey')
        ax4.spines['left'].set_linewidth(0.5)
        ax4.set_yticks([])
        ax4.tick_params(axis='x', rotation=45)
        if ylim is not None:
            ax4.set_ylim(ylim)

    # visuals
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_color('lightgrey')
    ax3.spines['top'].set_linewidth(0.5)
    # fig.tight_layout()

    # finalize
    if save_fig is not None:
        fig.savefig(save_fig, bbox_inches='tight')
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

    if hasattr(a, 'toarray'):  # scipy sparse matrices
        a = a.toarray()
    a = np.asarray(a)
    n_samples = prod(a.shape)
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
            warn(f"No good divisor found for magic reshaping {a.shape}. Filling with {m-n_samples}*{[fill_val]} to {m//best_divisor}x{best_divisor}.")
            a = np.pad(a, (0, m-n_samples), 'constant', constant_values=fill_val)
        # best_divisor = int_sqrt(n_samples)
        # if best_divisor**2 < n_samples*0.1:
        #     warn(f"No good divisor found for reshaping {a.shape} to a square. Using the best one: {best_divisor}x{n_samples//best_divisor}")
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
            warn("colorbar not supported for complex arrays. Use `complex_colorbar()` to see the color reference.")
        if cmap != 'hot':
            warn("Argument cmap is not used for complex arrays.")

    norm = None
    if vmin is not None or vmax is not None:
        if iscomplex:
            warn("vmin and vmax are not supported for complex arrays.")
        else:
            # check if one of the given values excludes all data
            if vmin is None:
                vmin = np.min(a)
            if vmax is None:
                vmax = np.max(a)
            if vmin > vmax:
                warn(f"vmin > vmax: {vmin} > {vmax}")
            norm = colors.Normalize(vmin, vmax)

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
            warn("colorbar not supported for RGB images.")
    else:
        raise ValueError(f"Array must be 2D or 1D, but shape was {a.shape}")

    def generate_ticks_and_labels(ticks, shape, max_ticks=10):
        def auto_round(x):
            if isinstance(x[0], float):
                return [f"{xi:.4g}" for xi in x]
            return ticks

        if isinstance(ticks, tuple) and len(ticks) == 2:
            if isinstance(ticks[0], (int, float)):
                ticklabels = np.linspace(np.min(ticks), np.max(ticks), min(max_ticks, shape))
                ticklabels = auto_round(ticklabels)
                ticks = np.linspace(0, shape-1, len(ticklabels))
            else:
                ticklabels = ticks[1]
                ticks = ticks[0]
        else:
            if len(ticks) > max_ticks:
                ticklabels = np.linspace(np.min(ticks), np.max(ticks), min(max_ticks, shape))
                ticklabels = auto_round(ticklabels)
            elif isinstance(ticks[0], float):
                ticklabels = auto_round(ticks)
            else:
                ticklabels = ticks
            ticks = np.linspace(0, shape-1, len(ticklabels))
        return ticks, ticklabels

    if isinstance(xticks, bool) and xticks == False:
        xticks = []
    if xticks is not None:
        xticks, xticklabels = generate_ticks_and_labels(xticks, a.shape[1])
        plt.xticks(xticks, xticklabels, rotation=xticks_rot, ha='center')
    if isinstance(yticks, bool) and yticks == False:
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
    plt.figure(figsize=(ceil(N/2),6))
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
            raise ValueError("Maxv must be higher than minv, but was: %f <= %f" % (maxv, minv))

        def getRGB(v):
            scaled = (v - minv)/(maxv - minv) * (len(colors)-1)
            scaled = max(0,min(scaled, len(colors)-1-1e-10)) #[0;len(colors)-1[
            subarea = floor(scaled)
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
            return "background-color: %s; color: %s" % (bg_color, color)

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
