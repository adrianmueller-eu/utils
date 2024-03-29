import os, time, sys
import numpy as np
from warnings import warn
from collections.abc import Iterable
from copy import deepcopy

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

def now():
    return time.time()


class ConvergenceCondition:
    """ Convergence condition for iterative algorithms.

    Additional properties
        x_prev (list): List of previous x's for error calculation. At most `converged_sequence_length` x's are stored.
        error (float): Error of the last iteration.
        start_time (float): Time of the first iteration.
        iter (int): Number of already performed iterations.
        skipped (int): Number of iterations the error was below eps.
        pbar (tqdm.tqdm): Progress bar object.
    """

    def __init__(self, max_iter=1000, eps=sys.float_info.epsilon, max_time=None, converged_sequence_length=2, skip_initial=0, skip_converged=0, use_tqdm=False):
        """ Convergence condition for iterative algorithms.

        Parameters
            max_iter (int): Maximum number of iterations. If `None`, no maximum is set.
            eps (float): Maximum error. If `None`, no error is calculated.
            max_time (float): Maximum time in seconds. If `None`, no maximum is set.
            converged_sequence_length (int): Number of previous iterations to check for convergence (useful for oscillating sequences)
            skip_initial (int): Number of iterations to let pass in the beggining before checking for convergence
            skip_converged (int): Number of iterations the error must be below eps
            use_tqdm (bool | tqdm.tqdm): Set `True` to show a `tqdm` progress bar. Give a `tqdm.tqdm` object to use a custom `tqdm` progress bar.

        Example:

            >>> def expm(A):
            >>>     i = 0
            >>>     res = A_i = np.eye(A.shape[0], dtype=A.dtype)
            >>>     has_converged = ConvergenceCondition()
            >>>     while not has_converged(res):
            >>>         i += 1
            >>>         A_i = A_i @ A / i
            >>>         res += A_i
            >>>     return res
            >>> expm(np.eye(2))
            array([[2.71828183, 0.        ],
                   [0.        , 2.71828183]])
        """
        self.max_iter = int(max_iter) if max_iter is not None else None
        self.eps = eps
        self.max_time = max_time
        self.converged_sequence_length = converged_sequence_length
        self.skip_initial = skip_initial
        self.skip_converged = skip_converged

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

        if max_iter is None and eps is None and max_time is None:
            raise ValueError("You must specify at least one of max_iter, eps, and max_time.")

    def __call__(self, x, iteration=None):
        if self.has_converged:
            return True

        self.has_converged = self.check_convergence(x, iteration)
        return self.has_converged

    @property
    def has_converged(self):
        return self.has_converged

    def check_convergence(self, x, iteration=None):
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
                    self.close()
                    return True
            else:
                self.skipped = 0
        if self.max_iter is not None:
            if self.iter >= self.max_iter:
                self.close()
                return True
        if self.max_time is not None:
            if self.start_time is None:
                self.start_time = time.time()
            if time.time() - self.start_time > self.max_time:
                self.close()
                return True

        # store x
        if self.x_prev is None:
            self.x_prev = [deepcopy(x)]
        else:
            if len(self.x_prev) >= self.converged_sequence_length:
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