import os
import numpy as np
from warnings import warn
from collections.abc import Iterable

T = True
F = False

def moving_avg(x, w=3):
    return np.convolve(x, np.ones(w), 'valid') / w

def r(x, precision=5):
    return np.round(x, precision)

def bins_sqrt(data):
    return int(np.ceil(np.sqrt(len(data))))

def logbins(data, start=None, stop=None, num=None, scale=2):
    if start is None:
        start = min(data)/scale
    if start <= 0:
        data = np.array(data)
        data_pos = data[data > 0]
        warn("Data set contains non-positive numbers (%.2f%%). They will be excluded for the plot." % (100*(1-len(data_pos)/len(data))))
        data = data_pos
        start = min(data)/scale
    if stop is None:
        stop = max(data)*scale
    if num is None:
        num = bins_sqrt(data)
    return 10**(np.linspace(np.log10(start),np.log10(stop),num))

def reversed_keys(d):
    return {k[::-1]:v for k,v in d.items()}

def npp(precision=5):
    np.set_printoptions(precision=5, suppress=True)

def mapp(func, *iterables, **kwargs):
    """map function that uses tq for progress bar"""
    for i in tq(range(len(iterables[0]))):
        yield func(*[iterable[i] for iterable in iterables], **kwargs)

def nbytes(n):
    """ Returns the number of bytes of some common objects. """
    # numpy array
    if hasattr(n, 'nbytes'):
        return n.nbytes
    # scipy sparse matrix
    elif hasattr(n, 'data') and isinstance(n.data, np.ndarray):
        return n.data.nbytes + n.indptr.nbytes + n.indices.nbytes
    # pandas dataframe
    elif hasattr(n, 'memory_usage'):
        return n.memory_usage(deep=True).sum()
    # pytorch tensor
    elif hasattr(n, 'element_size') and hasattr(n, 'storage'):
        return n.element_size() * n.storage().size()
    # string
    elif isinstance(n, str):
        return len(n)
    # list, tuple, set, dict, etc.
    elif isinstance(n, Iterable):
        return sum([nbytes(i) for i in n])
    else:
        raise TypeError(f"Can't get the size of an object of type {type(n)}")

def duh(n, precision=3):
    """ Takes a number of bytes and returns a human-readable string with the
    size in bytes, kilobytes, megabytes, or gigabytes.

    Parameters
        n (int | object): The number of bytes or an object the size of which to use (e.g. list, dict, numpy array, pandas dataframe, pytorch tensor)
        precision (int): The number of decimals to use
    """
    if not isinstance(n, int):
        n = nbytes(n)
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a positive integer, not {n}")

    for unit in ['B','KB','MB','GB','TB','PB','EB','ZB']:
        if n < 1024.0:
            break
        n /= 1024.0
    decimals = precision - int(n > 99) - int(n > 9) - int(n > 0)
    if decimals < 0 or unit == 'B':
        decimals = 0
    return f"{n:.{decimals}f} {unit}"

def shape_it(a, progress=True):
    """ Iterate over all indices of a numpy array. """
    from itertools import product
    from tqdm import tqdm as tq

    for n in tq(product(*[list(range(s)) for s in a.shape]),
                        disable=not progress, total=np.prod(a.shape)):
        yield n

def allclose_set(a, b):
    """ Check if for each item in a there is a corresponding item in b that is close to it and vice versa. """
    matched_b_indices = []
    for item_a in a:
        for i, item_b in enumerate(b):
            if i not in matched_b_indices and np.isclose(item_a, item_b):
                matched_b_indices.append(i)
                break
    return len(matched_b_indices) == len(a)

def tqt(iterable, **kwargs):
    if not isinstance(iterable, Iterable):
        raise TypeError(f"tqt expected an iterable, not {type(iterable)}")
    if len(iterable) == 0:
        raise ValueError("tqt expected a non-empty iterable")

    from tqdm.contrib import telegram

    token = os.environ['TELEGRAM_BOT_TOKEN']
    chat_id = os.environ['TELEGRAM_CHAT_ID']
    return telegram.tqdm(iterable, token=token, chat_id=chat_id, **kwargs)
