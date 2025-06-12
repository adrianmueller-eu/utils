import os, time, sys, numbers, warnings
import numpy as np
from collections.abc import Iterable
from collections import Counter
from copy import deepcopy
from math import log2, prod, floor, log10

def reversed_keys(d):
    return {k[::-1]:v for k,v in d.items()}

def nonunique(a, include_count=True):
    if include_count:
        return [(k, v) for k, v in Counter(a).most_common() if v > 1]
    return [k for k, v in Counter(a).items() if v > 1]

def mapl(func, *iterables, iterator=list, **kwargs):
    """map function that returns a collection (default: list)"""
    return iterator(map(func, *iterables, **kwargs))

def zipl(*iterables, iterator=list):
    """zip function that returns a collection (default: list)"""
    return iterator(zip(*iterables))

def rangel(*args, iterator=list, **kwargs):
    """range function that returns a collection (default: list)"""
    return iterator(range(*args), **kwargs)

class Slicable(type):
    def __getitem__(cls, idx):
        return cls(idx)[:]

class arange(metaclass=Slicable):
    """
    Range function that allows blockwise output.
    Example:

        arange(2, 12, 4)[:]
        Out: [[2, 3, 4, 5], [6, 7, 8, 9]]
    We can specify a step size *within* the blocks:

        arange(2, 12, 4, 2)[:]
        Out: [[2, 4], [6, 8]]
    Can also be used directly as a slice object to generate a range:

        arange[2:12:4]
        Out: [2, 6, 10]

    Parameters
    ----------
        start (int): Start of the range.
        stop (int): Stop of the range.
        block (int): Size of the blocks. If `None`, no blocks are created.
        step (int): Step size within the blocks. Default is 1.
        iterator (callable): Function to convert the range to a collection. Default is `list`.

    Returns
    -------
        list: Either a list of numbers, or a list of lists, where each inner list is a block of the range.
    """
    def __init__(self, start, stop=None, block=None, step=1, iterator=list):
        if iterator is not list:
            if iterator is np or iterator is np.ndarray or iterator is np.array:
                iterator = lambda x: np.array(list(x))
        if isinstance(start, slice):
            start, stop, step = start.start, start.stop, start.step
        elif stop is None:
            start, stop = 0, start
        self.start = start or 0
        self.stop = stop or 0
        self.step = step or 1
        self.block = block
        self.iterator = iterator
        if block is None:
            self._iter = range(self.start, self.stop, self.step)
        else:
            self._iter = range(self.start, self.stop - self.block + 1, self.block)

    def __getitem__(self, idx):
        it = self._iter[idx]
        if isinstance(it, range):
            if self.block is None:
                return self.iterator(it)
            return self.iterator(self.iterator(range(i, i + self.block, self.step)) for i in it)
        if self.block is None:
            return self.iterator([it])
        return self.iterator(range(it, it + self.block, self.step))

    def __str__(self):
        return f"arange({self.start}, {self.stop}, {self.block}, {self.step})"

    def __repr__(self):
        return self.__str__()

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

def timeit(fun, *args, runs=7, target=.01, max_loops=10**9):
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n): fun(*args)
        dt = time.perf_counter() - t0
        if dt >= target or n >= max_loops: break
        n *= 10

    samples = np.empty(runs)
    samples[0] = dt / n
    for i in range(1, runs):
        t0 = time.perf_counter()
        for _ in range(n): fun(*args)
        samples[i] = (time.perf_counter() - t0) / n
    return samples.mean(), samples.std(ddof=1), n * runs

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
    elif isinstance(o, (int)):  # huge ints
        return max(8, log2(o))
    elif o is None:
        return 0
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

def to_timestring(n, precision=4):
    """Takes a number of seconds and returns a human-readable string with the
    duration in y, d, h, min, s, ms, µs, or ns.

    Parameters
        n (int | float): The number of seconds
        precision (int): The number of significant digits to use in the output
    """
    assert n >= 0, f"n must be a positive number, but was: {n}"
    assert precision > 0, f"precision must be a positive number, but was: {precision}"

    units = [
        ('y', 31536000),      # 365.25 days
        ('d', 86400),
        ('h', 3600),
        ('min', 60),
        ('s', 1),
        ('ms', 1e-3),
        ('µs', 1e-6),
        ('ns', 1e-9),
    ]

    res = []
    for s, duration in units:
        if not res and n < duration and s != 'ns':
            continue
        qty = int(n//duration)
        if duration >= 60:
            if not res:
                precision -= floor(log10(qty)+1)
            elif s in ['h', 'min']:
                precision -= 2
            elif s == 'd':
                precision -= 3
            if precision <= 0:
                res += [[round(n/duration), 0, s]]
                break
            else:
                n %= duration
                res += [[qty, 0, s]]
        else:
            # ensure of these only one is printed
            if not res:
                if s != 'ns' or qty > 0:
                    precision -= floor(log10(qty)+1)
            elif s == 's':
                precision -= 2
            else:
                precision -= -floor(log10(duration)) + 3
            gap = np.inf if qty == 0 else floor(log10(qty)+1)
            if not res or precision + gap > 0:
                res += [[n/duration, max(precision, 0), s]]
            break

    # Carry over values rounded to a whole of the next higher unit
    unit_limits = {
        's': (60, 'min'),
        'min': (60, 'h'),
        'h': (24, 'd'),
        'd': (365, 'y'),
    }
    for i in range(len(res))[::-1]:
        resi0_displayed, unit = np.round(res[i][0], res[i][1]), res[i][2]
        if unit in unit_limits and resi0_displayed >= unit_limits[unit][0]:
            # This unit exceeds its limit
            carry, res[i][0] = divmod(resi0_displayed, unit_limits[unit][0])

            # Find the next unit up in our result
            next_idx = i - 1
            next_unit = unit_limits[unit][1]
            if next_idx >= 0 and res[next_idx][2] == next_unit:
                res[next_idx][0] += carry
            else:
                # Insert the next unit if it doesn't exist
                res.insert(next_idx + 1, [carry, 0, next_unit])

    return ' '.join(f'{q:.{prec}f}{s}' for q,prec,s in res)

def shape_it(shape, progress=False):
    """ Iterate over all indices of a numpy array. """
    from itertools import product
    if hasattr(shape, 'shape'):
        shape = shape.shape
    gen = product(*[list(range(s)) for s in shape])
    if progress:
        from tqdm.auto import tqdm as tq
        gen = tq(gen, total=prod(shape))
    return gen

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
    if hasattr(x, 'ndim'):
        if x.ndim != 0:
            return False
        x = x.item()
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

def as_list_not_str(a):
    if isinstance(a, list):
        return a
    if not is_iterable(a) or isinstance(a, str):
        a = [a]
    return list(a)

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

def warn(msg, category=UserWarning, stacklevel=2):
    """ Issue a warning with a custom stack level. """
    if isinstance(msg, str):
        warnings.warn(msg, category, stacklevel=stacklevel+1)
    else:
        warnings.warn(str(msg), category, stacklevel=stacklevel+1)
    sys.stdout.flush()  # flush potential other output first, too
    sys.stderr.flush()  # this flushes the warning output

def reissue_warnings(func):
    # Thanks to https://stackoverflow.com/questions/54399469/how-do-i-assign-a-stacklevel-to-a-warning-depending-on-the-caller
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record = True) as warning_list:
            result = func(*args, **kwargs)
        for warning in warning_list:
            warn(warning.message, warning.category)
        return result
    return inner

class Timer:
    def __init__(self, log=True):
        self._log = log
        self.restart()

    def restart(self):
        """Reset the initial and total time for the timer."""
        self.total = 0
        self.start_time = self._last_time = time.perf_counter()

    def done(self, task=""):
        """
        Call this method when a task is completed.
        Prints the message and the time elapsed since the last call.
        """
        elapsed = time.perf_counter() - self._last_time
        if self._log:
            print(f"Done {task} in " + to_timestring(elapsed))
        self.total += elapsed
        self._last_time = time.perf_counter()
        return elapsed

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
