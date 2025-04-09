import numpy as np

def binFrac_i(j, i):
    return int(j % (1/2**(i-1)) != j % (1/2**i))
    #return int(np.ceil((j % (1/2**(i-1))) - (j % (1/2**i))))

def binFrac(j, prec=20):
    return "." + "".join([str(binFrac_i(j,i)) for i in range(1,prec+1)])

def binstr_from_float(f, r=None, complement=False):
    """
    Convert a float `f` to a binary string with `r` bits after the comma.
    If `r` is None, the number of bits is chosen such that the float is
    represented exactly.

    Parameters
        f (float): The float to convert.
        r (int), optional: The number of bits after the comma. The default is None.
        complement (bool), optional: If True and `f < 0`, count the fraction "backwards" (e.g. -0.125 == '-.111').

    Returns
        str: The binary string representing `f`.
    """
    negative = f < 0
    if negative:
        f = -f # make it easier to handle the minus sign in the end
        if r is not None and r > 0 and abs(f) < 1/2**(r+1):
            return '.' + '0'*r
        if complement:
            # Translate the fraction to the corresponding complement, e.g. -0.125 => -0.875
            # alternatively, we could also flip all bits in `frac_part` below and add 1
            frac = f - int(f)
            if frac > 0:
                f = int(f) - frac + 1  # -1 if f was negative

    i = 0 # number of bits in the fraction part
    while int(f) != f:
        if r is not None and i >= r:
            f = int(np.round(f))
            break
        f *= 2
        i += 1
    f = int(f) # there should be no fractional part left

    # We use complement only for the fraction, not for the integer part
    # # If `f` is negative, the positive number modulus `2**k` is returned,
    # # where `k` is the smallest integer such that `2**k > -f`.
    # if f < 0:
    #     k = 0
    #     while -f > 2**(k-1):
    #         k += 1
    #     f = 2**k + f

    # integer part
    as_str = str(bin(f))[2:] # this adds a leading '-' sign for negative numbers
    sign = '-' if negative else ''
    # print(f, i, sign, as_str)
    if i == 0: # no fraction part
        if r is None or r <= 0: # ==> i == 0
            return sign + as_str
        if as_str == '0':
            return sign + '.' + '0'*r
        return sign + as_str + '.' + '0'*r
    int_part = sign + as_str[:-i]

    # fraction part
    frac_part = '0'*(i-len(as_str)) + as_str[-i:]
    # print(int_part, frac_part)
    if r is None:
       return int_part + '.' + frac_part
    return int_part + '.' + frac_part[:r] + '0'*(r-len(frac_part[:r]))

def float_from_binstr(s, complement=False):
    """ Convert a binary string to a float.

    Parameters
        s (str): The binary string.
        complement (bool, optional): If True, interpret the fraction part as the complement of the binary representation. Defaults to False.

    Returns
        float: The float represented by the binary string.
    """

    negative = s[0] == '-'
    if negative:
        s = s[1:]
    s = s.split('.')

    pre = 0
    frac = 0
    if len(s[0]) > 0:
        pre = int(s[0], 2)
    if len(s) > 1 and len(s[1]) > 0:
        if negative and complement:
            # flip all bits and add 1
            s[1] = ''.join(['1' if x == '0' else '0' for x in s[1]])
            frac = int(s[1], 2) + 1
        else:
            frac = int(s[1], 2)
        frac /= 2.**len(s[1])
    return float(pre + frac) * (-1 if negative else 1)

def binstr_from_int(n, places=0):
    if places > 0:
        if n < 0:
            return '-'+binstr_from_int(-n, places)
        res = f"{n:0{places}b}"
        if len(res) > places:
            raise ValueError(f"Integer {n} can't be represented in {places} bits")
        return res
    return f"{n:b}"

def int_from_binstr(s):
    return int(float_from_binstr(s))

def bincoll_from_binstr(s):
    return [int(x) for x in s]

def binstr_from_bincoll(l):
    return "".join([str(x) for x in l])

def int_from_bincoll(l):
    # return sum([2**i*v_i for i,v_i in enumerate(reversed(l))])
    return int_from_binstr(binstr_from_bincoll(l))
    # return 2**np.arange(len(l)) @ l[::-1]

def bincoll_from_int(n, places=0):
    return bincoll_from_binstr(binstr_from_int(n, places))

def count_bitreversed(q):
    if q <= 3:
        return np.array([int(bin(j)[2:].zfill(q)[::-1], 2) for j in range(2**q)])
    if q <= 8:
        bits = np.unpackbits(np.arange(2**q, dtype=np.uint8)).reshape(-1, 8)[:, :-q-1:-1]
        return 2**np.arange(q-1, -1, -1) @ bits.T
    return np.indices((2,)*q).reshape(q, -1).T @ (2**np.arange(q))
    # return sum([((np.arange(2**q) >> j) & 1) << (q - 1 - j) for j in range(q)])