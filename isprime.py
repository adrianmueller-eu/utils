from numpy.random import randint
from math import gcd, log

def is_prime(n, alpha=1e-20): # only up to 2^54 -> alpha < 1e-16.26 (-> 55 iterations; < 1e-20 is 67 iterations)
    """Miller-Rabin test for primality."""
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
        a = randint(2,n-2)
        if gcd(a,n) != 1:
            #print(n,"is not prime (1)")
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
                #print(n,"is not prime (2)")
                return False
        if gcd(b+1,n) == 1 or gcd(b+1,n) == n:
            p *= 1/2
        else:
            #print(n,"is not prime (3)")
            return False

    # print("%d is prime with alpha=%E (if Carmichael number: alpha=%f)" % (n, p, (3/4)**log(p,1/2)))
    return True

if __name__ == "__main__":
    import sys
    args = sys.argv
    n = int(args[-1])
    if is_prime(n):
      print("True")
    else:
      print("False")