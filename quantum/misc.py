import numpy as np
from functools import reduce

from ..plot import imshow_clean

def the_arrow(n=1, figsize=None):
    arrow = np.array([
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1]], dtype='i1')*2-1
    m = reduce(np.kron, [arrow]*n)
    if figsize is None:
        fs = max(1, 2**(2*n-4.68188))
        figsize = (fs, fs)
    imshow_clean(~m, figsize, cmap='hot')
