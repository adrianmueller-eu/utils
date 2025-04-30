import sys, warnings
import numpy as np

# e.g. series(lambda n, _: 1/factorial(2*n)) + series(lambda n, _: 1/factorial(2*n + 1))
def series(f, start_value=0, start_index=0, eps=sys.float_info.epsilon, max_iter=100000, verbose=False, tqdm=False):
    """ Calculate the series $start_value + \\sum_{n=start_index+1}^{\\infty} f(n, f(n-1, ...))$. Throws an error if the series doesn't converge.

    Parameters
        f (function): A function that takes two arguments, the current iteration `i` and the last term `term`, and returns the next term in the series.
        start_value (float | np.ndarray, optional): The value of the series at `start_index` (default: 0).
        start_index (int, optional): The index at which to start the series (default: 0).
        eps (float, optional): The precision to which the series should be calculated (default: `sys.float_info.epsilon`).
        max_iter (int, optional): The maximum number of iterations (default: 100000).
        verbose (bool, optional): If True, print the current iteration and the current value of the series (default: False).
        tqdm (tqdm.tqdm, optional): Use tqdm for progress bar (default: False). You might give a custom tqdm object.

    Returns
        float | np.ndarray: The value of the series.

    Examples:
        >>> series(lambda n, _: 1/factorial(2*n), 1) + series(lambda n, _: 1/factorial(2*n + 1), 1)
        2.7182818284590455
    """
    if not tqdm:
        def tq(x):  # dummy function
            return x
    elif not callable(tqdm):
        from tqdm.auto import tqdm as tq
    else:
        tq = tqdm

    if not np.isscalar(start_value):
        start_value = np.array(start_value, copy=True)
    res = start_value
    term = res
    for i in tq(range(start_index+1, max_iter)):
        term = f(i, term)
        res += term
        change = np.nansum(np.abs(term))
        if verbose:
            print(f"Iteration {i}:", res, term)
        if change < eps:
            return res  # return when converged
        if np.max(res) == np.inf:
            break

    raise ValueError(f"Series didn't converge after {max_iter} iterations! Error: {np.sum(np.abs(term))}")

def sequence(f, start_value=0, start_index=0, eps=sys.float_info.epsilon, max_iter=100000, verbose_n=0):
    """ Calculate the sequence $[f(start_index), f(start_index+1), ...]$ until it converges or the maximum number of iterations is reached. Then return the last term of the sequence.

    Parameters
        f (function): A function that takes two arguments, the current iteration `i` and last term `last_term`, and returns the next term in the sequence.
        start_value (float | np.ndarray, optional): The value of the series at `start_index` (default: 0).
        start_index (int, optional): The index at which to start the series (default: 0).
        eps (float, optional): The precision to which the series should be calculated (default: `sys.float_info.epsilon`).
        max_iter (int, optional): The maximum number of iterations (default: 100000).
        verbose_n (int, optional): If > 0, print the every n-th iteration and value of the series (default: 0).

    Returns
        float | np.ndarray: The value of the series.
    """
    if not np.isscalar(start_value):
        start_value = np.asarray(start_value)
    last_term = start_value
    for i in range(start_index+1, max_iter):
        current_term = f(i, last_term)
        if verbose_n > 0 and i % verbose_n == 0:
            print(f"Iteration {i}:", current_term)
        # if it contains inf or nan, we assume divergence
        if np.isinf(current_term).all() or np.isnan(current_term).all():
            warnings.warn(f"Sequence diverged after {i} iterations!", stacklevel=2)
            return current_term
        # if the difference between the last two terms is smaller than eps, we assume convergence
        error = np.sum(np.abs(current_term - last_term))
        if error < eps:
            if verbose_n > 0:
                print(f"Converged after {i} iterations! Error: {error}")
            return current_term
        last_term = current_term

    warnings.warn(f"Sequence didn't converge after {max_iter} iterations! Error: {error}", stacklevel=2)
    return current_term

def arctan2(y, x):  # same as np.arctan2
    if x == 0:
        return np.sign(y) * np.pi/2
    if y == 0:
        return (1 - np.sign(x)) * np.pi/2
    if x < 0:
        return np.arctan(y/x) + np.sign(y)* np.pi
    return np.arctan(y/x)
    # if x == 0:
    #     return np.sign(y) * np.pi/2
    # if x < 0:
    #     if y < 0:
    #         return np.arctan(y/x) - np.pi
    #     else:
    #         return np.arctan(y/x) + np.pi
    # return np.arctan(y/x)

def is_odd(x):
    return x % 2 == 1

def is_even(x):
    return x % 2 == 0

def deg(rad):
    return rad/np.pi*180

def rad(deg):
    return deg/180*np.pi

def softmax(a, beta=1):
    a = np.exp(beta*a)
    Z = np.sum(a)
    return a / Z

def choice(a, size=None, replace=True, p=None):
    if p is not None:
        if not isinstance(p, (int, float)):
            p_sum = np.sum(p)
            if np.abs(p_sum - 1) > 0.001:
                raise ValueError(f"Probabilities sum to {p_sum:.3f} ≠ 1")
        elif 0 <= p <= 1:
            p = [p, 1-p]
        else:
            raise ValueError(f"Invalid probability {p}")

    if hasattr(a, '__len__'):
        n = len(a)
        idx = np.random.choice(n, size=size, replace=replace, p=p)
        if isinstance(a, set):
            a = list(a)
        if isinstance(a, np.ndarray) or size is None:
            return a[idx]
        return [a[i] for i in idx]
    else:
        return np.random.choice(a, size=size, replace=replace, p=p)

def shuffle(a):
    if hasattr(a, 'copy'):
        a = a.copy()
    if isinstance(a, str):
        a = list(a)
        np.random.shuffle(a)
        return "".join(a)
    np.random.shuffle(a)
    return a

def gmean(a, axis=None):
    """ Geometric mean. """
    # ignore <= 0 and NaN values
    a = np.asarray(a)
    a = np.where(a <= 0, np.nan, a)
    return np.exp(np.nanmean(np.log(a), axis=axis))