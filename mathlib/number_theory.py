import numpy as np
from math import sqrt, ceil

sage_loaded = False
try:
    from sage.all import *
    sage_loaded = True
except ImportError:
    pass

def _two_arguments(f, *a):
    if len(a) == 2:
        return a
    elif len(a) == 3:
        return a[0], f(a[1], a[2])
    elif len(a) > 3:
        mid = len(a) // 2
        return f(*a[:mid]), f(*a[mid:])
    elif len(a) == 1:
        if hasattr(a[0], '__iter__'):
            return _two_arguments(f, *a[0])
        return a[0], None
    raise ValueError("No arguments given.")

def gcd(*a):
    """ Greatest common divisor. """
    a, b = _two_arguments(gcd, a)
    if hasattr(a, 'gcd'):
        return a.gcd(b)
    if not hasattr(a, '__bool__'):
        raise ValueError(f"Can't calculate gcd of {a} and {b}.")
    while b:
        a, b = b, a % b
    return a

def is_coprime(*a):
    """ Test if the given integers are coprime. """
    return gcd(*a) == 1

if not sage_loaded:

    def is_prime(n, alpha=1e-20): # only up to 2^54 -> alpha < 1e-16.26 (-> 55 iterations; < 1e-20 is 67 iterations)
        """ Miller-Rabin test for primality. """
        if n == 1 or n == 4:
            return False
        if n == 2 or n == 3:
            return True

        def getKM(n):
            k = 0
            while n % 2 == 0:
                k += 1
                n /= 2
            return k,int(n)

        p = 1
        while p > alpha:
            a = np.random.randint(2,n-2)
            if gcd(a,n) != 1:
                # print(n,"is not prime (1)", f"gcd({a},{n}) = {gcd(a,n)}")
                return False
            k,m = getKM(n-1)
            b = pow(a, m, n)
            if b == 1:
                p *= 1/2
                continue
            for i in range(1,k+1):
                b_new = pow(b,2,n)
                # first appearance of b == 1 is enough
                if b_new == 1:
                    break
                b = b_new
                if i == k:
                    # print(n,"is not prime (2)")
                    return False
            if gcd(b+1,n) == 1 or gcd(b+1,n) == n:
                p *= 1/2
            else:
                # print(n,"is not prime (3)")
                return False

        # print("%d is prime with alpha=%E (if Carmichael number: alpha=%f)" % (n, p, (3/4)**log(p,1/2)))
        return True

    def prime_factors(n):
        """ Simple brute-force algorithm to find prime factors. """
        factors = []
        # remove all factors of 2 first
        while n % 2 == 0:
            factors.append(2)
            n //= 2
        i = 3
        while i * i <= n:
            if n % i:
                i += 2
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors

    def euler_phi(n):
        """ Euler's totient function. """
        return round(n * np.prod([1 - 1/p for p in prime_factors(n)]))
        # return sum([1 for k in range(1, n) if is_coprime(n, k)])  # ~100x slower

    def lcm(*a):
        """ Least common multiple. """
        a, b = _two_arguments(lcm, a)
        if b is None:
            return a
        if hasattr(a, 'lcm'):
            return a.lcm(b)
        return a * b // gcd(a, b)

    def next_prime(a):
        """ Find the next prime number after `a`. """
        if a < 2:
            return 2
        a = int(a) + 1
        while not is_prime(a):
            a += 1
        return a

def closest_prime_factors_to(n, m):
    """Find the set of prime factors of n with product closest to m."""
    if not sage_loaded:
        from .sets import powerset

    pf = prime_factors(n)

    min_diff = float("inf")
    min_combo = None
    for c in powerset(pf):
        diff = abs(m - np.prod(c))
        if diff < min_diff:
            min_diff = diff
            min_combo = c
    return min_combo

def int_sqrt(n):
    """ For integer $n$, find the integer $a$ closest to $\\sqrt{n}$, such that $n/a$ is also an integer. """
    if n == 1 or n == 0:
        return n
    return int(np.prod(closest_prime_factors_to(n, sqrt(n))))

def next_good_int_sqrt(n, p=0.1):
    """ For integer $n$, look for the next larger integer $m$, such that `int_sqrt(m)**2` is between `p*m` and `1/p*m`. """
    assert 0 < p < 1
    # try n first
    if p*n <= int_sqrt(n)**2 < 1/p*n:
        return n
    m = n + 2 if n % 2 == 0 else n + 1  # skip primes
    while not p*m <= int_sqrt(m)**2 < 1/p*m:
        m += 2
    return m

def dlog(x,g,n):
    """ Discrete logarithm, using the baby-step giant-step algorithm.

    Parameters
        x (int): The number to find the logarithm of.
        g (int): The base.
        n (int): The modulus.

    Returns
        int: The discrete logarithm of `x` to the base `g` modulo `n`.
    """
    w = ceil(sqrt(euler_phi(n)))
    baby = []
    for j in range(w+1):
        baby.append(x*(g**j) % n)
#     print(baby)
    for i in range(1, w+1):
        giant = pow(g, i*w, n)
#         print(i, giant)
        if giant in baby:
            return i * w-baby.index(giant) % n

    raise ValueError(f"Couldn't find discrete logarithm of {x} to the base {g} modulo {n}.")

def is_carmichael(n):
    """ Test if `n` is a Carmichael number. """
    if is_prime(n): # are neither even nor prime
        return False
    for a in range(2,n):
        if gcd(n,a) == 1 and pow(a,n-1,n) != 1:
            return False
    return True

def carmichael_numbers(to):
    for n in range(3*5*7, to, 2): # have at minimum three prime factors and not even
        if is_carmichael(n):
            yield n