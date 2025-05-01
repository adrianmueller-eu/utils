import psutil, warnings
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

from .utils import count_qubits, transpose_qubit_order, verify_subsystem, partial_trace

from ..utils import is_int, is_iterable, duh, is_from_assert, shape_it
from ..mathlib.matrix import normalize, is_hermitian, is_psd, random_vec, trace_product, generate_recursive, su, commutes, is_diag, eigh, outer
from ..mathlib import binstr_from_int, softmax, choice
from ..plot import colorize_complex
from ..prob import random_p, check_probability_distribution

def state_trace(state, retain_qubits, reorder=True):
    """This is a pervert version of the partial trace, but for state vectors. I'm not sure about the physical 
    meaning of its output, but it was at times helpful to visualize and interpret subsystems, especially when 
    the density matrix was out of reach (or better: out of memory)."""
    state = np.asarray(state)
    state[np.isnan(state)] = 0
    n = count_qubits(state)

    # sanity checks
    if not hasattr(retain_qubits, '__len__'):
        retain_qubits = [retain_qubits]
    if len(retain_qubits) == 0:
        retain_qubits = range(n)
    elif n == len(retain_qubits):
        return state, np.abs(state)**2
    elif max(retain_qubits) >= n:
        raise ValueError(f"No such qubit: %d" % max(retain_qubits))

    state = state.reshape(tuple([2]*n))
    probs = np.abs(state)**2

    cur = 0
    for i in range(n):
        if cur not in retain_qubits:
            state = np.sum(state, axis=cur)
            probs = np.sum(probs, axis=cur)
            # shift unvisited qubits to the left
            retain_qubits = [q-1 if q > cur else q for q in retain_qubits]
        else:
            cur += 1

    state = state.reshape(-1)
    state = normalize(state) # renormalize

    probs = probs.reshape(-1)
    assert np.abs(np.sum(probs) - 1) < 1e-5, np.sum(probs) # sanity check

    if reorder:
        state = transpose_qubit_order(state, retain_qubits)
        probs = transpose_qubit_order(probs, retain_qubits)

    return state, probs

def plotQ(state, showqubits=None, showcoeff=True, showprobs=True, showrho=False, figsize=None, title=""):
    """My best attempt so far to visualize a state vector. Control with `showqubits` which subsystem you're 
    interested in (`None` will show the whole state). `showcoeff` utilitzes `state_trace`, `showprobs` shows 
    a pie chart of the probabilities when measured in the standard basis, and `showrho` gives a plt.imshow view 
    on the corresponding density matrix."""

    def tobin(n, places):
        return ("{0:0" + str(places) + "b}").format(n)

    def plotcoeff(ax, state):
        n = count_qubits(state)
        if n < 6:
            basis = [tobin(i, n) for i in range(2**n)]
            #plot(basis, state, ".", figsize=(10,3))
            ax.scatter(basis, state.real, label="real")
            ax.scatter(basis, state.imag, label="imag")
            ax.tick_params(axis="x", rotation=45)
        elif n < 9:
            ax.scatter(range(2**n), state.real, label="real")
            ax.scatter(range(2**n), state.imag, label="imag")
        else:
            ax.plot(range(2**n), state.real, label="real")
            ax.plot(range(2**n), state.imag, label="imag")

        #from matplotlib.ticker import StrMethodFormatter
        #ax.xaxis.set_major_formatter(StrMethodFormatter("{x:0"+str(n)+"b}"))
        ax.legend()
        ax.grid()

    def plotprobs(ax, state):
        n = count_qubits(state)
        toshow = {}
        cumsum = 0
        for idx in probs.argsort()[-20:][::-1]: # only look at 20 largest
            if cumsum > 0.96 or probs[idx] < 0.01:
                break
            toshow[tobin(idx, n)] = probs[idx]
            cumsum += probs[idx]
        if np.abs(1-cumsum) > 1e-15:
            toshow["rest"] = max(0,1-cumsum)
        ax.pie(toshow.values(), labels=toshow.keys(), autopct=lambda x: f"%.1f%%" % x)

    def plotrho(ax, rho):
        n = count_qubits(rho)
        rho = colorize_complex(rho)
        ax.imshow(rho)
        if n < 6:
            basis = [tobin(i, n) for i in range(2**n)]
            ax.set_xticks(range(rho.shape[0]), basis)
            ax.set_yticks(range(rho.shape[0]), basis)
            ax.tick_params(axis="x", rotation=45)

    state = ket(state, renormalize=False)

    # trace out unwanted qubits
    if showqubits is None:
        n = count_qubits(state)
        showqubits = range(n)

    if showrho:
        memory_requirement = (len(showqubits))**2 * 16
        #print(memory_requirement / 1024**2, "MB") # rho.nbytes
        if memory_requirement > psutil.virtual_memory().available:
            raise ValueError(f"Too much memory required ({duh(memory_requirement)}) to calulate the density matrix!")
        rho = partial_trace(state, retain_qubits=showqubits, reorder=True)

    state, probs = state_trace(state, showqubits, reorder=True)

    if showcoeff and showprobs and showrho:
        if figsize is None:
            figsize=(16,4)
        fig, axs = plt.subplots(1,3, figsize=figsize)
        fig.subplots_adjust(right=1.2)
        plotrho(axs[0], rho)
        plotcoeff(axs[1], state)
        plotprobs(axs[2], state)
    elif showcoeff and showprobs:
        if figsize is None:
            figsize=(18,4)
        fig, axs = plt.subplots(1,2, figsize=figsize)
        fig.subplots_adjust(right=1.2)
        plotcoeff(axs[0], state)
        plotprobs(axs[1], state)
    elif showcoeff and showrho:
        if figsize is None:
            figsize=(16,4)
        fig, axs = plt.subplots(1,2, figsize=figsize)
        fig.subplots_adjust(right=1.2)
        plotcoeff(axs[0], state)
        plotrho(axs[1], rho)
    elif showprobs and showrho:
        if figsize is None:
            figsize=(6,4)
        fig, axs = plt.subplots(1,2, figsize=figsize)
        plotrho(axs[0], rho)
        plotprobs(axs[1], state)
    else:
        fig, ax = plt.subplots(1, figsize=figsize)
        if showcoeff:
            plotcoeff(ax, state)
        elif showprobs:
            plotprobs(ax, state)
        elif showrho:
            plotrho(ax, rho)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

def random_ket(n, size=None, kind='fast'):
    """ Sample Haar-random state vectors ($2^{n+1}-1$ degrees of freedom).
    - `kind='fast'` generates state vectors by sampling each element from a normal distribution, resulting in sampling from the Haar measure.
    - `kind='ortho'` generates a set of orthonormal states using QR decomposition. At most $2^n$ orthogonal states can be generated (in the last batch dimension).
    """
    assert is_int(n) and n >= 1, f"Invalid number of qubits: n should be an integer >= 1, but was: {n}"
    if not is_iterable(size):
        size = (size,)
    assert size == (None,) or all(s >= 1 for s in size), f"size should contain only integers >= 1, but was: {size}"
    if kind == 'fast' or size == None:
        if size == (None,):
            return normalize(random_vec(2**n, complex=True, kind='normal'))
        kets = random_vec((*size, 2**n), complex=True, kind='normal')
        return normalize(kets, axis=-1)
    elif kind == 'ortho':
        if len(size) > 1:
            res = np.empty(size + (2**n,), dtype=complex)
            for i in shape_it(size[:-1]):
                res[i] = random_ket(n, size[-1], kind='ortho')
            return res
        size_ = size[0] or 1
        assert size_ <= 2**n, f"Can't generate more than 2**{n} = {2**n} orthogonal states, but requested was: {size}"
        # inspired by the method to sample Haar-random unitaries
        kets = random_vec((2**n, min(size_, 2**n)), complex=True, kind='normal')
        Q, R = np.linalg.qr(kets)
        Rd = np.diag(R)
        L = Rd / np.abs(Rd)
        kets = L[:,None] * Q.T
        if size == (None,):
            return kets[0]
        return kets
    else:
        raise ValueError(f"Unknown kind: {kind}")

def random_dm(n=1, rank='full'):
    """
    Generate a random density matrix ($rank*2^{n+1}-1$ degrees of freedom).
    `rank` can be an integer between 1 and 2^n, or one of 'pure' or 'full'.
    """
    assert is_int(n), f"n needs to be an integer, but was: {n}"
    if rank == 'pure':
        rank = 1
    if rank == 'full':
        rank = 2**n
    assert is_int(rank) and rank >= 1, f"rank should be an integer >= 1, but was: {rank}"
    assert rank <= 2**n, f"A {n}-qubit density matrix can be at most rank {2**n}, but requested was: {rank}"

    if rank == 1:
        state = random_ket(n)
        return outer(state)

    kets = random_ket(n, rank, kind='ortho')
    probs = random_p(len(kets), kind='uniform')
    return kets.conj().T @ (probs[:,None] * kets)

def ket_from_int(d, n=None):
    if not n:
        n = int(np.ceil(np.log2(d+1))) or 1
    elif d >= 2**n:
        raise ValueError(f"A {n}-qubit state doesn't have basis state {d} (max is {2**n-1})")
    # return np.array(bincoll_from_int(2**d, 2**n)[::-1], dtype=float)
    res = np.zeros(2**n)
    res[d] = 1
    return res

def ket(specification, n=None, renormalize=True, check=1):
    """Convert a string or dictionary of strings and weights to a state vector. The string can be a binary number 
    or a combination of binary numbers and weights. The weights will be normalized to 1."""
    # if a string is given, convert it to a dictionary
    if isinstance(specification, (np.ndarray, list, tuple)):
        psi = np.asarray(specification)
        if psi.ndim == 1 and renormalize:
            psi_norm = np.linalg.norm(psi)
            psi /= psi_norm
            assert_ket(psi, n, check=0)  # norm is already checked
        assert_ket(psi, n, check=check)
        return psi
    if is_int(specification):
        return ket_from_int(specification, n)
    if type(specification) == str:
        if specification == "random":
            return random_ket(n or 1)
        # handle some special cases: |+>, |->, |i>, |-i>
        if specification in ["+", "-", "i", "-i", "0", "1"]:
            if n == None:
                n = 1
            if specification == "+":
                return np.ones(2**n)/2**(n/2)
            elif specification == "-":
                s = [1,-1]
            elif specification == "i":
                s = [1,1j]
            elif specification == "-i":
                s = [1,-1j]
            elif specification == "0":
                s = np.zeros(2**n)
                s[0] = 1
                return s
            elif specification == "1":
                s = np.zeros(2**n)
                s[-1] = 1
                return s
            else:
                raise ValueError(f"WTF-Error: {specification}")
            s = normalize(s)
            return reduce(np.kron, [s]*n).astype(complex)
        # if a string of + and - only
        if set(specification) <= {"+", "-"}:
            if n is None:
                n = len(specification)
            plus  = normalize([1,1])
            minus = normalize([1,-1])
            return reduce(np.kron, [plus if s == "+" else minus for s in specification]).astype(complex)

        # remove whitespace
        specification = specification.replace(" ", "")
        specification_dict = dict()

        # Parse the specification into the dictionary, where the keys are the strings '00', '01', '10', '11', etc. and the values are the weights
        # The following cases have to be considered:
        #  00 + 11
        #  00 - 11
        #  00 - 0.1*11
        #  0.5*00 + 0.5*11
        #  0.5*(00 + 11)
        #  (1+1j)*00 + (1-1j)*11
        #  0.5*((1+1j)*00 + (1-1j)*11)
        #  0.5*((1+1j)*00 + (1-1j)*11) + 0.5*((1-1j)*00 + (1+1j)*11)

        # if there are no brackets, then split by "+" and then by "*"
        if "(" not in specification and ")" not in specification:
            specification = specification.replace("-", "+-")
            for term in specification.split("+"):
                if term == "":
                    continue
                if "*" in term:
                    weight, state = term.split("*")
                    weight = complex(weight)
                elif term[0] == '-':
                    weight = -1
                    state = term[1:]
                else:
                    weight = 1
                    state = term
                if n is None:
                    n = len(state)
                else:
                    assert len(state) == n, f"Part of the specification has wrong length for {n} qubits: len('{state}') != {n}"
                if state in specification_dict:
                    specification_dict[state] += weight
                else:
                    specification_dict[state] = weight

            specification = specification_dict
        else:
            raise NotImplementedError("Parentheses are not yet supported!")

    # convert the dictionary to a state vector
    n = len(list(specification.keys())[0])
    state = np.zeros(2**n, dtype=complex)
    for key in specification:
        state[int(key, 2)] = specification[key]
    return normalize(state)

def unket(state, as_dict=False, prec=5, check=1):
    """ Reverse of `ket`.

    `prec` serves as filter for states close to 0, and if `as_dict==False`, it also defines to which precision 
    the values are rounded in the string.

    Example:
    >>> unket(ket('00+01+10+11'))
    '0.5*(00+01+10+11)'
    >>> unket(ket('00+01+10+11'), as_dict=True)
    {'00': 0.5, '01': 0.5, '10': 0.5, '11': 0.5}
    """
    eps = 10**(-prec) if prec is not None else 0
    state = ket(state, renormalize=False, check=check)
    n = count_qubits(state)
    if as_dict:
        # cast to float if imaginary part is zero
        if np.all(np.abs(state.imag) < 1e-12):
            state = state.real
        return {binstr_from_int(i, n): state[i] for i in range(2**n) if np.abs(state[i]) > eps}

    if prec is not None:
        state = np.round(state, prec)
    if n == 0:
        return str(state[0])
    # group by weight
    weights = {}
    for i in np.where(np.abs(state) > eps)[0]:
        weight = state[i]
        pre = ''
        # Find a close value in weights
        for w in weights:
            if abs(w - weight) < eps:
                weight = w
                break
            if abs(w + weight) < eps:
                weight = w
                pre = '-'
                break
        else:
            # remove imaginary part if it's zero
            if np.abs(weight.imag) < eps:
                weight = weight.real
            # create new weight
            weights[weight] = []
        # add state to weight
        weights[weight].append(pre + binstr_from_int(i, n))

    # convert to string
    res = []
    for weight in weights:
        if weight == 1:
            res += weights[weight]
        elif len(weights[weight]) == 1:
            res += [f"{weight}*{weights[weight][0]}"]
        else:
            res += [f"{weight}*({'+'.join(weights[weight])})"]
    return "+".join(res).replace("+-", "-")

def op(specification1, specification2=None, n=None, check=1):
    """
    Generate an operator from two state vectors. If only one state vector is given, it will be used for both.
    """
    if specification2 is None:
        s2 = s1 = ket(specification1, n=n, renormalize=True, check=check)
    else:
        if is_int(specification1) and is_int(specification2):
            n = n or int(np.ceil(np.log2(max(specification1, specification2)+1))) or 1
        s1 = ket(specification1, n=n, renormalize=True, check=check)
        s2 = ket(specification2, n=n or count_qubits(s1), renormalize=True, check=check)

    return outer(s1, s2.conj())

def dm(kets, p=None, n=None, renormalize=True, check=3):
    """
    Generate a density matrix from a list of state vectors and probabilities.
    """
    if isinstance(kets, (list, np.ndarray)):
        kets = np.asarray(kets)
        assert kets.ndim < 3, f"Invalid shape for state vectors: {kets.shape}"
        if kets.ndim == 2:
            if p is None:
                assert kets.shape[0] == kets.shape[1], f"More than 1 ket given, but no probabilities"
                rho = kets
                if renormalize:
                    rho_tr = np.trace(rho)
                    if not abs(rho_tr - 1) < 1e-8:
                        rho = rho / rho_tr
                assert_dm(rho, n, check=check)
                return rho
            assert kets.shape[0] == p.shape[0] or kets.shape[1] == p.shape[0], f"Compatible shapes for state vectors and probabilities: {kets.shape} <≠> {p.shape}"
            p = check_probability_distribution(p, check=check)
            if kets.shape[0] != p.shape[0]:
                kets = kets.T
            if check >= 1:
                for k in kets:
                    assert_ket(k, n)
            rho = kets @ (p[:,None] * kets.conj().T)
            return rho
    elif isinstance(kets, str):
        if kets in ['random', 'random_dm']:
            return random_dm(n or 1, rank='full')
        elif kets == 'random_pure':
            kets = 'random'
    elif n is None and is_int(p):
        n = p

    psi = ket(kets, n, renormalize=renormalize, check=check)
    return outer(psi)

def as_state(state, renormalize=True, n=None, check=2):
    if not check:
        return state
    try:
        return ket(state, n=n, renormalize=renormalize, check=check)
    except:
        pass
    return dm(state, n=n, renormalize=renormalize, check=check)

def ev(obs, state, check=2):
    if check >= 2:
        assert is_hermitian(obs)
    if state.ndim == 1:
        assert_ket(state, check=check)
        return (state.conj() @ obs @ state).real
    assert_dm(state, check=check)
    return trace_product(obs, state).real

def probs(state):
    """ Probabilities of outcomes (vector or density matrix) when measuring in the standard basis."""
    state = np.asarray(state)
    if state.ndim == 2:
        return np.diag(state).real
    return np.abs(state)**2

def is_ket(psi, n=None, print_errors=True, check=1):
    """Check if `ket` is a valid state vector."""
    return is_from_assert(assert_ket, print_errors)(psi, n, check)

def assert_ket(psi, n=None, check=1):
    """ Check if `ket` is a valid state vector. """
    if isinstance(psi, str):
        try:
            psi = ket(psi)
        except Exception as e:
            assert False, f"Invalid state vector: {psi}"
    elif isinstance(psi, int):
        return n is None or psi < 2**n
    try:
        psi = np.asarray(psi)
    except Exception as e:
        assert False, f"Invalid state vector: {psi}"
    assert np.issubdtype(psi.dtype, np.number)  # norm is faster for float than complex
    n = n or count_qubits(psi)
    assert len(psi.shape) == 1 and psi.shape[0] == 2**n, f"Invalid state vector shape: {psi.shape} ≠ {(2**n,)}"
    if check >= 1:
        assert abs(np.linalg.norm(psi) - 1) < 1e-10, f"State vector is not normalized: {np.linalg.norm(psi)}"

def is_dm(rho, n=None, print_errors=True, check=3):
    """Check if matrix `rho` is a density matrix."""
    return is_from_assert(assert_dm, print_errors)(rho, n, check)

def assert_dm(rho, n=None, check=3):
    """ Check if matrix `rho` is a density matrix. """
    if isinstance(rho, str):
        try:
            rho = dm(rho, renormalize=False, check=0)
        except Exception as e:
            assert False, f"Invalid density matrix: {rho}"
    elif isinstance(rho, int):
        return n is None or rho < 2**n
    try:
        rho = np.asarray(rho, dtype=complex)  # float is no performance difference in np.trace
    except Exception as e:
        assert False, f"Invalid density matrix: {rho}"
    assert len(rho.shape) == 2 and rho.shape[0] == rho.shape[1], f"Invalid density matrix shape: {rho.shape}"
    n = n or count_qubits(rho)
    assert rho.shape[0] == 2**n, f"Invalid density matrix shape: {rho.shape} ≠ {(2**n, 2**n)}"
    assert abs(np.trace(rho) - 1) < 1e-10, f"Density matrix is not normalized: {np.trace(rho)}"
    assert is_psd(rho, check=check), "Density matrix is not positive semi-definite"

def is_state(state, n=None, print_errors=True, check=3):
    """ Check if `state` is a valid state vector or density matrix. """
    return is_from_assert(assert_state, print_errors)(state, n, check=check)

def assert_state(state, n=None, check=3):
    """ Check if `state` is a valid state vector or density matrix. """
    try:
        assert_ket(state, n=n, check=check)
    except AssertionError:
        assert_dm(state, n=n, check=check)

def is_pure_dm(rho, check=3, tol=1e-12):
    """ Check if matrix `rho` is a pure density matrix. """
    if not is_dm(rho, check=check):
        return False
    # return np.linalg.matrix_rank(rho) == 1
    if check >= 1:
        return abs(trace_product(rho, rho) - 1) < tol
    return True

def is_separable_state(state, subsystem, n=None, tol=1e-12, check=3):
    """
    Check if the state is separable with respect to the given qubits.
    """
    state = as_state(state, n=n, check=check)
    n = n or count_qubits(state)
    subsystem = verify_subsystem(subsystem, n)
    q = len(subsystem)
    if q == 0 or q == n:
        return True

    # if ket, check if the rdm is pure
    if state.ndim == 1:
        # use the smaller subsystem
        if q > n - q:
            subsystem = [q for q in range(n) if q not in subsystem]
            q = n - q
        rdm = partial_trace(state, subsystem)
        return is_pure_dm(rdm, tol=tol)

    assert state.ndim == 2, f"Invalid state shape: {state.shape}"
    # if dm, check if all branches are the same
    # use smaller subsystem -> more branches -> faster failure
    if q > n - q:
        subsystem = [q for q in range(n) if q not in subsystem]
    state = transpose_qubit_order(state, subsystem, reshape=True)
    # last block is reference
    B = state[:,-1,:,-1]
    # find a nonzero diagonal element in B (all nonnegative and real by definition)
    idx = None
    for i in range(len(B)):  # n-q x n-q block
        if B[i,i] > 1e-7:
            idx = (i,i)
            break
    assert idx is not None, f"Could not find a nonzero diagonal element in: {B}"
    B_ref = B[idx]

    # check if the first block is the same as B
    B00 = state[:,0,:,0]
    s = B_ref/B00[idx]
    if not np.allclose(s*B00, B, atol=tol):
        return False

    # faster than svd or iterating over all blocks if it is separable
    state = state.reshape(2**n, -1)
    rdm = partial_trace(state, subsystem)
    rdm2 = partial_trace(state, [q for q in range(n) if q not in subsystem])
    rdm_full = np.kron(rdm, rdm2)
    return np.allclose(rdm_full, state, atol=tol)

def ket_from_dm(rho, kind='sample', tol=0, check=3):
    """ Convert a density matrix `rho` to a state vector. """
    rho = dm(rho, check=check)
    if kind == 'sample':
        probs, kets = eigh(rho)
        # filter out small eigenvalues
        mask = probs >= tol
        probs = probs[mask]
        kets = kets.T[mask]
        return choice(kets, p=probs)
    elif kind == 'max':
        probs, kets = np.linalg.eigh(rho)
        return kets.T[np.argmax(probs)]
    else:
        raise ValueError(f"Unknown kind: {kind}")

def dm_from_ensemble(probs, kets, check=2):
    check_probability_distribution(probs, check=check)
    if check >= 2:
        for k in kets:
            assert_ket(k)
    return sum(p * outer(k) for p, k in zip(probs, kets))

def ensemble_from_state(rho, filter_eps=1e-10, check=3):
    rho = as_state(rho, check=check)
    if rho.ndim == 1:
        return np.array([1.]), np.array([rho])
    return ensemble_from_dm(rho, filter_eps=filter_eps, check=check)

def ensemble_from_dm(rho, filter_eps=1e-10, check=3):
    if check:
        rho = dm(rho, check=check)
    n = count_qubits(rho)
    if n > 5 and is_diag(rho, tol=0):
        probs = np.diag(rho).real
        kets = np.eye(2**n, dtype=complex)
        return probs, kets
    probs, kets = eigh(rho)
    # filter out zero eigenvalues
    mask = probs > filter_eps
    probs = probs[mask]
    kets = kets.T[mask]
    return probs.real, kets

def is_eigenstate(psi, H, tol=1e-10, check=2):
    if len(psi.shape) == 2:
        rho = dm(psi, check=check)
        return commutes(rho, H, tol)
    psi = ket(psi, renormalize=True)  # non-normalized states can still be eigenstates
    psi2 = H @ psi
    eigval_abs = np.linalg.norm(psi2)
    if eigval_abs < tol:
        return True
    return abs(abs(psi.conj() @ psi2) - eigval_abs) < tol

def gibbs(H, beta=1, check=2):
    """Calculate the Gibbs state of a Hamiltonian `H` at inverse temperature `beta`."""
    H = np.asarray(H)
    if check >= 2:
        assert is_hermitian(H), "Hamiltonian must be Hermitian"
    assert beta >= 0, f"Inverse temperature must be positive, but was {beta}"
    E, U = eigh(H)
    E = softmax(E, -beta)
    return U @ (E[:,None] * U.conj().T)

Wigner_A = None
def _init_Wigner_A():
    global Wigner_A
    if Wigner_A is None:
        I, X, Y, Z = su(2, True)
        Wigner_A = [
            0.5 * (I + X + Y + Z),
            0.5 * (I - X - Y + Z),
            0.5 * (I + X - Y - Z),
            0.5 * (I - X + Y - Z)
        ]
    return Wigner_A

def get_Wigner_A(i,j,n):
    Wigner_A = _init_Wigner_A()
    i_s = [int(x) for x in f'{i:0{n}b}']
    j_s = [int(x) for x in f'{j:0{n}b}']
    return reduce(np.kron, [Wigner_A[2*i_s[k] + j_s[k]] for k in range(n)])

def Wigner_matel_from_state(state, i, j):
    state = np.asarray(state)
    n = count_qubits(state)
    A = get_Wigner_A(i,j,n)
    if state.ndim == 1:
        return (state.conj() @ A @ state).real / 2**n
    return trace_product(state, A).real / 2**n

def Wigner_from_state(state, check=2):
    state = as_state(state, check=check)
    n = count_qubits(state)
    if n > 8:
        warnings.warn(f"Generating {2**(2*n)} {2**n}x{2**n} matrices (n = {n}) may take too a long time.", stacklevel=2)
    W = np.zeros((2**n, 2**n))
    isket = state.ndim == 1
    Wigner_A = _init_Wigner_A()
    for idx, A in enumerate(generate_recursive(Wigner_A, n, Wigner_A, np.kron)):
        base4 = f'{idx:0{2*n}b}'
        i, j = int(base4[::2], 2), int(base4[1::2], 2)
        if isket:
            W[i,j] = (state.conj() @ A @ state).real / 2**n
        else:
            W[i,j] = trace_product(state, A).real / 2**n
    return W

def dm_from_Wigner(W):
    """ The density matrix can be reconstructed from the Wigner matrix since all information is conserved (4^n - 1 dofs). """
    n = count_qubits(W)
    rho = np.zeros_like(W, dtype=complex)
    for i, j in shape_it(W):
        rho += W[i,j] * get_Wigner_A(i,j,n)
    return rho

## TODO: def random_stabilizer_state(n): + convert n to full quantum state
## TODO: random_tensor_network_state + convert tensor network state -> full quantum state