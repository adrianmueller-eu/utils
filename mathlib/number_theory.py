import re
import numpy as np
from math import sqrt, ceil, prod, log

sage_loaded = False
try:
    from sage.all import *
    sage_loaded = True
except ImportError:
    from .sets import powerset

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

def smallest_prime_factor(n, start=3):
    """ Find the smallest prime factor of n (n if prime), starting from the **odd** integer `start`. """
    if n < 2:
        return None
    if n % 2 == 0:
        return 2
    # assert start % 2 == 1, f"start must be odd, but was {start}"
    for i in range(start, int(n**0.5) + 1, 2):
        if n % i == 0:
            return i
    return n

def is_prime_brute(n):
    """ Simple brute-force algorithm to check for primality. """
    return n == smallest_prime_factor(n)

def Miller_Rabin(n, alpha=1e-20): # only up to 2^54 -> alpha < 1e-16.26 (-> 55 iterations; < 1e-20 is 67 iterations)
    """ Miller-Rabin test for primality. """
    if n == 1 or n == 4:
        return False
    if n == 2 or n == 3:
        return True

    def getKM(n):
        k = 0
        while n % 2 == 0:
            k += 1
            n //= 2
        return k, n

    p = 1
    while p > alpha:
        a = np.random.randint(2,n-2)
        if gcd(a,n) != 1:
            # print(n,"is not prime (1)", f"gcd({a},{n}) = {gcd(a,n)}")
            return False
        k, m = getKM(n-1)
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
        gcdb1n = gcd(b+1,n)
        if gcdb1n == 1 or gcdb1n == n:
            p *= 1/2
        else:
            # print(n,"is not prime (3)")
            return False

    # print("%d is prime with alpha=%E (if Carmichael number: alpha=%f)" % (n, p, (3/4)**log(p,1/2)))
    return True

def is_prime_regex(n):
    """ Using regex `/^1?$|^(11+?)\1+$/` to check for primality. See https://stackoverflow.com/questions/2795065/how-to-determine-if-a-number-is-a-prime-with-regex """
    return not re.match(r'^1?$|^(11+?)\1+$', '1'*n)

if not sage_loaded:

    def is_prime(n):
        if n < 3*10**8:
            return is_prime_brute(n)
        return Miller_Rabin(n)

    def prime_factors(n):
        """ Simple algorithm to find prime factors. """
        if n < 2:
            return []
        factors, p, n = [], 2, int(n)
        while True:
            p = smallest_prime_factor(n, start=max(3, p))
            factors.append(p)
            if p == n:
                break
            n //= p
        return factors

    def euler_phi(n):
        """ Euler's totient function. """
        return round(n * prod([1 - 1/p for p in set(prime_factors(n))]))
        # return sum([1 for k in range(1, n) if is_coprime(n, k)])  # ~100x slower

    def euler_phi_lower(n):
        """Asymptotic lower bound for Euler's totient function: $n/(e^gamma * log(log(n)))$ """
        return n / (np.euler_gamma * log(log(n)))

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
        if a % 2 == 0:
            a += 1
        while not is_prime(a):
            a += 2
        return a

    def primes(to, start=2):
        """ Generate all prime numbers from `start` to `to`. """
        res = []
        if start <= 2:
            res.append(2)
            start = 3
        else:
            start = int(start)
            if start % 2 == 0:
                start += 1
        if to is None:
            to = start + 10**20
        for a in range(start, int(to), 2):
            if is_prime(a):
                res.append(a)
        return res

def divisors(n):
    """ Find all divisors of `n`. """
    pf = prime_factors(n)
    return set([prod(c) for c in powerset(pf)])

def closest_prime_factors_to(n, m):
    """Find the set of prime factors of n with product closest to m."""
    min_diff = float("inf")
    min_combo = None
    for d in divisors(n):
        diff = abs(m - d)
        if diff < min_diff:
            min_diff = diff
            min_combo = d
    return prime_factors(min_combo)

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

    Parameters:
        x (int): The number to find the logarithm of.
        g (int): The base.
        n (int): The modulus.

    Returns:
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

class Group:
    def __init__(self, elements, identity=None):
        self.elements = elements
        self._identity = identity

    def op(self, x, y):
        raise NotImplementedError()

    def inv(self, x):
        if type(self).pow != Group.pow:
            return self.pow(x, -1)
        for y in self.elements:
            if np.all(self.op(x, y) == self.identity):
                return y
        raise ValueError(f"Couldn't find inverse of {x} in group {self}")

    def pow(self, x, n):
        if n == 0:
            return self.identity
        assert np.isclose(n, int(n)), "Exponent must be an integer"
        n = int(n)
        if n < 0:
            x = self.inv(x)
            n = -n
        while n % 2 == 0:
            x = self.op(x, x)
            n //= 2
        tmp = x
        for _ in range(n):
            x = self.op(x, tmp)
        return x

    @property
    def id(self):
        return self.identity
    @property
    def identity(self):
        if self._identity is not None:
            return self._identity
        for x in self.elements:
            for y in self.elements:
                if not np.all(self.op(x, y) == y):
                    break
            else:
                self._identity = x
                return x
        raise ValueError("Couldn't find an identity element!")

    def order(self, x=None):
        if x is not None:
            assert x in self, f"{x} not in group {self}"
            ds = divisors(len(self))
            for d in ds:
                if self.pow(x, d+1) == x:
                    return d
        if callable(x):
            return np.inf
        return len(self)

    def is_generator(self, x):
        return self.order(x) == len(self)

    def generators(self):
        return [g for g in self if self.is_generator(g)]

    def is_cyclic(self):
        for x in self:
            if self.is_generator(x):
                return True
        return False

    def center(self):
        center = set()
        for x in self:
            for y in self:
                if not np.all(self.op(x, y) == self.op(y, x)):
                    break
            else:
                center.add(x)
        return center

    def is_abelian(self):
        for x in self.elements:
            for y in self.elements:
                if not np.all(self.op(x, y) == self.op(y, x)):
                    return False
        return True

    def __getitem__(self, x):
        return self.elements[x]

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __contains__(self, x):
        return x in self.elements

    def __repr__(self):
        return repr(self.elements)

class Zn_star(Group):
    """ Multiplicative group of integers modulo `n`. """
    def __init__(self, n):
        self.n = n
        if is_prime(n):
            els = range(1, n)
        else:
            els = [i for i in range(n) if is_coprime(i, n)]
        super().__init__(els, 1)

    def op(self, x, y):
        return (x * y) % self.n

    def pow(self, x, n):
        return pow(x, n, self.n)

    def is_cyclic(self):
        if self.n in [1, 2, 4]:
            return True
        pf = prime_factors(self.n)
        if pf[0] == 2:
            return pf[1] != 2 and len(set(pf)) == 2
        return len(set(pf)) == 1

    def is_field(self):
        return len(self) == self.n - 1
        # return is_prime(self.n)

    def __repr__(self):
        res = f"Z_{self.n}^* ({len(self.elements)} elements)"
        if isinstance(self.elements, range) or len(self.elements) < 100:
            res += f": {self.elements}"
        else:
            res += f": {self.elements[:100]}"
            return res[:-1] + ", ...]"
        return res