import numpy as np
from math import factorial, log2, log, sqrt

from .basic import series

golden_ratio = (1 + sqrt(5))/2
def Fibonacci(n):
    Psi = 1 - golden_ratio
    return int(np.round((golden_ratio**n - Psi**n)/(golden_ratio - Psi)))  # /sqrt(5)

def calc_pi1(prec=100):
    """ Calculate pi using the Gauss-Legendre algorithm. """
    from decimal import Decimal, getcontext
    getcontext().prec = prec # int(np.e*2**N-2)
    N = int(log2(prec + 2))
    a = Decimal(1)
    b = Decimal(1)/Decimal(2).sqrt()
    t = Decimal(1)/Decimal(4)
    p = Decimal(1)
    for n in range(N):
        a_next = (a+b)/2
        b_next = (a*b).sqrt()
        t_next = t - p*(a-a_next)**2
        p_next = 2*p
        a = a_next
        b = b_next
        t = t_next
        p = p_next
    return (a+b)**2/(4*t)

def calc_pi2(prec=100):
    """ Calculate pi using the Chudnovsky algorithm. """
    from decimal import Decimal, getcontext
    getcontext().prec = prec  # int(14*N)
    N = int(prec/14)
    r = Decimal(0)
    for n in range(N):
        r_i = Decimal(factorial(6*n)*(13591409+545140134*n))/Decimal(factorial(3*n)*(factorial(n)*(-640320)**n)**3)
        r += r_i
    return Decimal(4270934400)/(Decimal(10005).sqrt()*r)

def calc_pi3(prec=100):
    """ Calculate pi using the Ramanujan algorithm. """
    from decimal import Decimal, getcontext
    getcontext().prec = prec  # int(8*N)
    N = int(prec/7)
    r = Decimal(0)
    for n in range(N):
        r_i = Decimal(factorial(4*n)*(1103+26390*n))/Decimal((factorial(n)*396**n)**4)
        r += r_i
    return Decimal(9801)/(Decimal(8).sqrt()*r)

def calc_pi4(prec=100):
    """ Calculate pi using the Bailey-Borwein-Plouffe algorithm. """
    # return BBP_formula(1, 16, 8, [4, 0, 0, -2, -1, -1])
    from decimal import Decimal, getcontext
    getcontext().prec = prec  # int(1.25*N+2.5)
    N = int((prec-2.5)/1.2)
    r = Decimal(0)
    for n in range(N):
        r_i = Decimal(1)/Decimal(16**n)*(Decimal(4)/(8*n+1)-Decimal(2)/(8*n+4)-Decimal(1)/(8*n+5)-Decimal(1)/(8*n+6))
        r += r_i
    return r

def BBP_formula(s, b, m, a, N=100, verbose=False):
    """ Calculate the sum of the series $sum_{k=0}^N 1/(b^k) (sum_{j=1}^m a_j/(mk+j)^s)$.
    This is a general form to calculate irrational numbers, e.g.
    - `np.pi == BBP_formula(1, 16, 8, [4, 0, 0, -2, -1, -1])`
    - `np.log(a) - np.log(a-1) == 1/2*BBP_formula(1, a, 1, [1])`

    See also https://en.wikipedia.org/wiki/Bailey–Borwein–Plouffe_formula
    """
    def P(k, _):
        return 1/(b**k)*sum([a[j]/(m*k+j+1)**s for j in range(len(a))])

    return series(P, start_value=0, start_index=-1, max_iter=N+1, verbose=verbose)

def log_(x, base=np.e):
    """
    Calculating the logarithm by foot: Use that $\\ln(1+x) = \\int_0^x \\frac 1{1+t} dt = \\sum_{n=0}^\\infty (-1)^n \\frac{x^{n+1}}{n+1}$. 
    However, this series converges only for $|x| < 1$ and quickly only for $|x| < 1/2$. So, divide x by e until 0.5 < x' < 1.5.
    Then use that ln(e^k*x') = k + ln(x').
    Speedup by using 2 instead of e (easier divison / multiplication), and that $ln(1+x) - ln(1-x)$ only contains terms with even powers (faster convergence).
    """
    if base != np.e:
        return log_(x)/log_(base)
    if x < 0:
        return np.nan
    if x == 0:
        return -np.inf
    k = 0
    ln2 = log(2)
    if x > 0.5:
        while np.abs(x - 1) >= 0.5:
            x /= 2
            k += 1
    else:
        while np.abs(x - 1) >= 0.5:
            x *= 2
            k -= 1
    # x_ = x - 1
    # lnx_ = series(lambda n, _: (-1)**n * x_**(n+1)/(n+1), start_index=-1)
    x_ = (x-1)/(x+1)
    lnx_ = series(lambda n, _: 2/(2*n+1) * x_**(2*n+1), start_index=-1)
    return k*ln2 + lnx_

log_table, log_table_keys = None, None
def log_2(x, base=np.e):
    """
    Calculating the logarithm using Feynman's algorithm. Performance gets close (~ factor 5) to np.log.
    """
    if base != np.e:
        return log_2(x)/log_2(base)
    if x < 0:
        return np.nan
    if x == 0:
        return -np.inf

    global log_table, log_table_keys
    if log_table is None:
        log_table, log_table_keys = [], []
        # build the table
        for k in range(52+1):  # 52-bit precision in 64-bit float
            k2k = 1 + 2**(-k)
            log_table_keys.append(k2k)
            log_table.append(log(k2k))

    # bring x in range 1 <= x < 2
    k = 0
    if x > 1:
        while x >= 2:
            x /= 2
            k += 1
    else:
        while x < 1:
            x *= 2
            k -= 1

    # calculate ln(x) by composing x as factors 1 + 2^{-k}
    x_ = 1
    lnx_ = 0
    for j in range(1,53):
        x__ = x_ * log_table_keys[j]
        if x__ < x:
            x_ = x__
            lnx_ += log_table[j]

    return k*log_table[0] + lnx_

### more

def sqrt_brain_compatible(x, correction_term=False, n_max = 20):
    """ Nice way to calculate approximate square roots in the head:
        1. Find the largest integer n, such that n² < x
        2. sqrt(x) ≈ n + (x-n²)/(2n+1) + 1/(6(2n+1))
    The last term is more negligible the larger x.
        """
    n_sq_table = [n**2 for n in range(1,n_max)]
    if x <= 1:
       raise ValueError("Are you kidding me")
    if x > n_sq_table[-1]:
       raise ValueError("Use a proper function for this, please.")
    for n,n2 in enumerate(n_sq_table):
       if n2 > x:
           if correction_term:
               return n + (x - n_sq_table[n-1])/(2*n + 1) + 1/(6*(2*n+1))
           else:
               return n + (x - n_sq_table[n-1])/(2*n + 1)

# Reduction of the halting problem to the equivalence problem, i.e. show that the latter is at least as hard as the former.
# Give an algorithm `H` and input `x` to decide whether `H(x)` halts. This function returns two functions `f1, f2`. Use your
# implementation `equiv` solving the equivalence problem to solve the halting problem, e.g. `equiv(*equiv_from_halt(H, x))`.
def equiv_from_halt(H, x):
    def f1(y):
        if y == x:
            return H(x)
        return True

    def f2(y):
        return True

    return f1, f2

def Hutchinson_trace(A, n=1000):
    """ Terribly inefficient, but theoretically interesting. """
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    zs = np.random.normal(0, 1, size=(A.shape[0], n))
    return np.mean([z.T @ A @ z for z in zs.T])
