import os, time, sys, numbers, warnings
import numpy as np
from warnings import warn
from collections.abc import Iterable
from copy import deepcopy
from math import log2

def reversed_keys(d):
    return {k[::-1]:v for k,v in d.items()}

def mapl(func, *iterables, iterator=list, **kwargs):
    """map function that returns a collection (default: list)"""
    return iterator(map(func, *iterables, **kwargs))

def zipl(*iterables, iterator=list):
    """zip function that returns a collection (default: list)"""
    return iterator(zip(*iterables))

def rangel(*args, iterator=list, **kwargs):
    """range function that returns a collection (default: list)"""
    return iterator(range(*args), **kwargs)

def tqmap(func, *iterables, **kwargs):
    """map function that uses tq for progress bar"""
    from tqdm.auto import tqdm as tq

    for i in tq(range(len(iterables[0]))):
        yield func(*[iterable[i] for iterable in iterables], **kwargs)

def tqmapl(func, *iterables, iterator=list, **kwargs):
    """map function that uses tq for progress bar and returns a collection (default: list)"""
    return iterator(tqmap(func, *iterables, **kwargs))

def tqt(iterable, **kwargs):
    if not is_iterable(iterable):
        raise TypeError(f"tqt expected an iterable, not {type(iterable)}")
    if len(iterable) == 0:
        raise ValueError("tqt expected a non-empty iterable")

    from tqdm.contrib import telegram

    token = os.environ['TELEGRAM_BOT_TOKEN']
    chat_id = os.environ['TELEGRAM_CHAT_ID']
    return telegram.tqdm(iterable, token=token, chat_id=chat_id, **kwargs)

def print_gen(gen):
    for x in gen:
        print(x)

def print_header(title, char='#'):
    res = "\n"
    charl = len(char)
    res += char*(len(title)//charl + 2*4) + "\n"
    res += char*3 + ' '*(charl - len(title) % charl) + title + ' '*charl + char*3 + "\n"
    res += char*(len(title)//charl + 2*4)
    print(res)

def nbytes(o):
    """ Returns the number of bytes of some common objects. """
    # numpy array
    if hasattr(o, 'nbytes'):
        return o.nbytes
    # scipy sparse matrix
    elif hasattr(o, 'data') and isinstance(o.data, np.ndarray):
        total = o.data.nbytes
        if hasattr(o, 'indptr'):
            total += o.indptr.nbytes
        if hasattr(o, 'indices'):
            total += o.indices.nbytes
        if hasattr(o, 'rows'):
            total += o.rows.nbytes
        return total
    # pandas dataframe
    elif hasattr(o, 'memory_usage'):
        return o.memory_usage(deep=True).sum()
    # pytorch tensor
    elif hasattr(o, 'element_size') and hasattr(o, 'storage'):
        return o.element_size() * o.storage().size()
    # string
    elif isinstance(o, str):
        return len(o)
    # dict
    elif isinstance(o, dict):
        return sum(nbytes(k) + nbytes(v) for k, v in o.items())
    # list, tuple, set
    elif is_iterable(o):
        try:
            o = np.asarray(o)
            if np.issubdtype(o.dtype, np.number):
                return o.nbytes
        finally:
            return sum(nbytes(i) for i in o)
    # numeric types
    elif isinstance(o, (float)):
        return 8
    elif isinstance(o, (complex)):
        return 16
    elif isinstance(o, (bool)):
        return 4
    elif isinstance(o, (int)):
        return max(8, log2(o))
    else:
        raise TypeError(f"Can't get the size of an object of type {type(o)}")

def duh(n, precision=3):
    """ Takes a number of bytes and returns a human-readable string with the
    size in bytes, kilobytes, megabytes, or gigabytes.

    Parameters
        n (int | object): The number of bytes or an object the size of which to use (e.g. list, dict, numpy array, pandas dataframe, pytorch tensor)
        precision (int): The number of decimals to use
    """
    if not is_int(n) or n > 1 << 63:
        n = nbytes(n)
    if not is_int(n) or n < 0:
        raise ValueError(f"n must be a positive integer, not {n}")

    for unit in ['B','KB','MB','GB','TB','PB','EB','ZB']:
        if n < 1024.0:
            break
        n /= 1024.0
    decimals = precision - int(n > 99) - int(n > 9) - int(n > 0)
    if decimals < 0 or unit == 'B':
        decimals = 0
    return f"{n:.{decimals}f} {unit}"

def shape_it(shape, progress=False):
    """ Iterate over all indices of a numpy array. """
    from itertools import product
    if hasattr(shape, 'shape'):
        shape = shape.shape
    gen = product(*[list(range(s)) for s in shape])
    if progress:
        from tqdm.auto import tqdm as tq
        gen = tq(gen, total=np.prod(shape))

    for n in gen:
        yield n

def size_samples(f, size):
    """ Create a numpy array of shape `size, ...` by sampling using the function `f`. """
    if size is None or size == ():
        return f()
    if is_iterable(size):
        size = tuple(size)
    if not isinstance(size, tuple):
        size = (size,)
    assert all(s >= 1 for s in size), f"size should contain only integers >= 1, but was: {size}"

    el = np.asarray(f())
    if size == (1,):
        return np.array([el])
    res = np.empty(size + el.shape, dtype=el.dtype)
    for idx in shape_it(size):
        res[idx] = f()
    return res

def last(gen):
    """ Get the last element of a generator. """
    for x in gen:
        pass
    return x

def is_int(x):
    # https://stackoverflow.com/questions/3501382/checking-whether-a-variable-is-an-integer-or-not
    try:
        return int(x) == x
    except:
        if isinstance(x, complex):
            return int(x.real) == x.real and x.imag == 0
        return False

def is_numeric(x):
    return isinstance(x, numbers.Number)

def is_iterable(x):
    return isinstance(x, Iterable)

def is_from_assert(func, print_error=True):
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
            return True
        except AssertionError as e:
            if print_error:
                warnings.warn(f"AssertionError: {e}", stacklevel=3)
            return False
    return inner

def as_callable(x):
    if callable(x):
        return x
    else:
        return lambda *args, **kwargs: x

def startfile(filepath):
    import subprocess, os, platform
    if platform.system() == 'Darwin':       # macOS
        return subprocess.call(('open', filepath))
    elif platform.system() == 'Windows':    # Windows
        return os.startfile(filepath)
    else:                                   # linux variants
        return subprocess.call(('xdg-open', filepath))

def reissue_warnings(func):
    # Thanks to https://stackoverflow.com/questions/54399469/how-do-i-assign-a-stacklevel-to-a-warning-depending-on-the-caller
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record = True) as warning_list:
            result = func(*args, **kwargs)
        for warning in warning_list:
            warnings.warn(warning.message, warning.category, stacklevel = 2)
        return result
    return inner

class ConvergenceCondition:
    def __init__(self, max_iter=1000, eps=sys.float_info.epsilon, max_value=None, max_time=None, period=2, skip_initial=0, skip_converged=0, use_tqdm=False, verbose=True):
        """ Convergence condition for iterative algorithms.

        Parameters
            max_iter (int):              Maximum number of iterations.
            eps (float):                 Minimum error (i.e. terminates if error `eps` is reached. See also `skip_converged`.) If `None`, no error is calculated.
            max_value (float):           Maximum value in the sequence (useful to check for divergence)
            max_time (float):            Maximum time in seconds. If `None`, no maximum is set.
            period (int):                Number of previous iterations to check for convergence (useful for oscillating sequences)
            skip_initial (int):          Number of iterations to let pass in the beggining before checking for convergence
            skip_converged (int):        Number of successive iterations the error must be below `eps`
            use_tqdm (bool | tqdm.tqdm): Set `True` to show a `tqdm` progress bar. Give a `tqdm.tqdm` object to use a custom `tqdm` progress bar.
            verbose (bool):              Set `True` to print the reason for termination.

        Additional properties
            has_converged (bool): Whether any convergence condition has been met.
            iter (int):           Number of already performed iterations.
            error (float):        Error of the last iteration.
            x_prev (list):        List of previous x's for error calculation. At most `period` x's are stored.
            start_time (float):   Clock time of the first iteration.
            skipped (int):        Number of successive iterations the error was below `eps`.
            pbar (tqdm.tqdm):     Progress bar object.

        Example:

            >>> def expm(A):
            >>>     res = A_i = np.eye(A.shape[0], dtype=A.dtype)
            >>>     conv = ConvergenceCondition()
            >>>     while not conv(res):
            >>>         A_i = A_i @ A / conv.iter
            >>>         res += A_i
            >>>     return res
            >>> expm(np.eye(2))
            array([[2.71828183, 0.        ],
                   [0.        , 2.71828183]])
        """
        self.max_iter = int(max_iter) if max_iter is not None else None
        self.eps = eps
        self.max_value = max_value
        self.max_time = max_time
        self.period = period
        self.skip_initial = skip_initial
        self.skip_converged = skip_converged
        self.verbose = verbose

        self.x_prev = None
        self.error = None
        self.start_time = None
        self.iter = 0
        self.skipped = 0

        self.has_converged = False

        try:
            import tqdm
            import tqdm.auto

            if isinstance(use_tqdm, tqdm.tqdm):
                self.pbar = use_tqdm
            elif use_tqdm:
                self.pbar = tqdm.auto.tqdm(total=self.max_iter)
            else:
                self.pbar = None
        except ImportError:
            self.pbar = None
            if use_tqdm:
                warn("tqdm is not installed. Install it with `pip install tqdm` to use the progress bar.")

        if max_iter is None and eps is None and max_value is None and max_time is None:
            raise ValueError("You must specify at least one of `max_iter`, `eps`, `max_value`, and `max_time`.")

    def __call__(self, x, iteration=None):
        if self.has_converged:
            return True

        self.has_converged = self._check_convergence(x, iteration)
        return self.has_converged

    def _check_convergence(self, x, iteration=None):
        if iteration is not None:
            self.iter = iteration
        else:
            self.iter += 1
        if self.pbar is not None:
            if iteration is not None:
                self.pbar.update(iteration - self.pbar.n)
            else:
                self.pbar.update(1)
        if self.iter <= self.skip_initial:
            return False

        if self.eps is not None and self.x_prev is not None:
            # calculate error with respect to all previous x's and take the min
            self.error = min([np.linalg.norm(x - x_prev) for x_prev in self.x_prev])
            if self.error < self.eps:
                if self.skipped < self.skip_converged:
                    self.skipped += 1
                    return False
                else:
                    if self.verbose:
                        print(f"Converged at iteration {self.iter} with error {self.error}")
                    self.close()
                    return True
            else:
                self.skipped = 0
        if self.max_iter is not None:
            if self.iter >= self.max_iter:
                if self.verbose:
                    print(f"Reached maximum number of iterations {self.max_iter}")
                self.close()
                return True
        if self.max_value is not None:
            if np.max(np.abs(x)) >= self.max_value:
                if self.verbose:
                    print(f"Maximum value {self.max_value} (index {np.argmax(abs(x))}) at iteration {self.iter}")
                self.close()
                return True
        if self.max_time is not None:
            if self.start_time is None:
                self.start_time = time.time()
            if time.time() - self.start_time > self.max_time:
                if self.verbose:
                    print(f"Time limit reached at iteration {self.iter}")
                self.close()
                return True

        # store x
        if self.x_prev is None:
            self.x_prev = [deepcopy(x)]
        else:
            if len(self.x_prev) >= self.period:
                self.x_prev = self.x_prev[1:]
            self.x_prev.append(deepcopy(x))
        return False

    def close(self):
        if self.pbar is not None:
            self.pbar.close()

    def reset(self):
        self.x_prev = None
        self.error = None
        self.start_time = None
        self.iter = 0
        self.skipped = 0
        if self.pbar is not None:
            self.pbar.reset()

    def __str__(self):
        end_time_fmt = time.strftime("%H:%M:%S", time.localtime(self.start_time + self.max_time)) if self.start_time is not None else None
        return f"ConvergenceCondition(max_iter={self.max_iter}, eps={self.eps}, max_time={self.max_time}) at iteration {self.iter} with error {self.error} and end time {end_time_fmt}"

    def __repr__(self):
        return str(self)
