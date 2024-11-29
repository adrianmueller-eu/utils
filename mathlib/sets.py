import numpy as np
from itertools import chain, combinations

sage_loaded = False
try:
    from sage.all import *
    sage_loaded = True
except ImportError:
    pass

if not sage_loaded:
    # https://docs.python.org/3/library/itertools.html
    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def bipartitions(iterable, unique=False):
    """ All bipartitions of an iterable.
    >>> list(bipartitions([0,1,2], unique=True))
    >>> [([0], [1, 2]), ([1], [0, 2]), ([0, 1], [2])]
    """
    s = list(iterable)
    n = len(s)
    end = 1 << (n-1) if unique else (1 << n) - 1
    for i in range(1, end):
        part1 = [s[j] for j in range(n) if (i >> j) & 1]
        part2 = [s[j] for j in range(n) if not (i >> j) & 1]
        if unique:
            yield (part1, part2) if part1 < part2 else (part2, part1)
        else:
            yield part1, part2

def allclose_set(a, b):
    """ Check if for each item in a there is a corresponding item in b that is close to it and vice versa. """
    # matched_b_indices = []
    # for item_a in a:
    #     for i, item_b in enumerate(b):
    #         if i not in matched_b_indices and np.isclose(item_a, item_b):
    #             matched_b_indices.append(i)
    #             break
    # return len(matched_b_indices) == len(a)
    # convert to numpy arrays if they are not
    if isinstance(a, set):
        a = list(a)
    a = np.sort(np.reshape(a, -1))
    if isinstance(b, set):
        b = list(b)
    b = np.sort(np.reshape(a, -1))
    # check if they have the same length
    if len(a) != len(b):
        return False
    # check if they are close
    return np.allclose(a, b)