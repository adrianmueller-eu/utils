from warnings import warn
import numpy as np
from math import sqrt, ceil, log10
import matplotlib.pyplot as plt

def moving_avg(x, w=3):
    return np.convolve(x, np.ones(w), 'valid') / w

def bins_sqrt(data):
    return ceil(sqrt(len(data)))

def logbins(data, start=None, stop=None, num=None, scale=2):
    if start is None:
        start = min(data)/scale
    if start <= 0:
        data = np.asarray(data)
        data_pos = data[data > 0]
        warn("Data set contains non-positive numbers (%.2f%%). They will be excluded for the plot." % (100*(1-len(data_pos)/len(data))))
        data = data_pos
        start = min(data)/scale
    if stop is None:
        stop = max(data)*scale
    if num is None:
        num = bins_sqrt(data)
    return 10**(np.linspace(log10(start),log10(stop),num))

# make a list out of a pd.corr() matrix
def corrList(corr, index_names=("feature 1", "feature 2")):
    import pandas as pd

    corr = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
    corr = pd.DataFrame(corr.stack(), columns=["correlation"])
    corr.index.names = index_names
    return corr

def plot_dendrogram(X, method="ward", truncate_after=25, metric='euclidean', ax=None):
    from scipy.cluster.hierarchy import dendrogram, linkage
    Z = linkage(X, metric=metric, method=method)
    dendrogram(Z, truncate_mode='lastp', p=truncate_after, leaf_rotation=90, ax=ax)

def plot_dendrograms(X, methods=("single", "complete", "average", "centroid", "ward"), truncate_after=50, metric='euclidean'):
    fig, axes = plt.subplots(len(methods), figsize=(12,6*len(methods)))
    for ax, method in zip(axes, methods):
        ax.set_title(method)
        plot_dendrogram(X, method, truncate_after, metric, ax)

# from scipy.stats import circmean
def circular_mean(X, mod=360):
    rads = X*2*np.pi/mod
    av_sin = np.mean(np.sin(rads))
    av_cos = np.mean(np.cos(rads))
    av_rads = np.arctan2(av_sin,av_cos) % (2*np.pi) # move the negative half to the other side -> [0;2pi]
    return av_rads * mod/(2*np.pi)

# todo: data whitening, remove nan, intelligent data cleaning, automatic outlier detection

# use with scipy.optimize.minimize and numpy arrays
def sqloss(f,x,y):
    def _sqloss(p):
        res = f(x,*p) - y
        return res.T @ res
    return _sqloss

def noise(size=None, eps=0.1, kind='gaussian'):
    if kind == 'gaussian':
        return np.random.normal(0, eps, size)
    elif kind == 'uniform':
        return np.random.uniform(-eps, eps, size)
    else:
        raise ValueError(f"Unknown noise kind: {kind}")

# # association mining
# from mlxtend.frequent_patterns import apriori, association_rules
# res = pd.DataFrame(index=df.index)
#
# def s(name, series):
#     res[name] = series.astype(int)
#
# # create binarization here
# s("class_0", df["class"] == 0)
#
# res.head()
#
# # find frequent itemsets
# min_support = 0.1
# frequent = apriori(res, min_support=min_support, use_colnames=True)
# # and take a look at them
# print(len(frequent))
# frequent.sort_values("support", ascending=False).reset_index(drop=True).head(10)
#
# # find rules
# min_confidence = 0.5
# rules = association_rules(frequent, metric='confidence', min_threshold=min_confidence)
#
# # and look at them
# # %matplotlib
# rules.plot("support", "confidence", kind="scatter", c="lift", cmap="viridis",
#     xlim=(min_support-0.02,1.02), ylim=(min_confidence-0.02,1.02), figsize=(10,6))
# plt.grid()
#
# rules.sort_values("lift", ascending=False).head(10)
