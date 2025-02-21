import psutil
import numpy as np
import matplotlib.pyplot as plt
from math import log2
from functools import reduce
try:
    from scipy.linalg import eigh
except:
    from numpy.linalg import eigh

from ..utils import is_int, is_iterable, duh, is_from_assert, shape_it
from ..mathlib import normalize, binstr_from_int, is_hermitian, softmax, is_psd, random_vec, trace_product
from ..plot import colorize_complex
from ..prob import random_p, check_probability_distribution

def transpose_qubit_order(state, new_order, assume_square=True):
    state = np.asarray(state)
    n = count_qubits(state)

    # infer batch shape
    if len(state.shape) == 2 and state.shape[0] == state.shape[1]:  # state is not necessarily a density matrix or ket (e.g. hamiltonian, unitary)
        if assume_square:
            batch_shape = ()
        else:  # kets
            batch_shape = state.shape[:1]
    elif len(state.shape) >= 2 and state.shape[-2] != state.shape[-1]:
        batch_shape = state.shape[:-1]
    elif len(state.shape) >= 3 and state.shape[-2] == state.shape[-1]:
        batch_shape = state.shape[:-2]
    else:
        batch_shape = ()
    batch_shape = list(batch_shape)

    # parse new_order
    if isinstance(new_order, int) and new_order == -1:
        new_order = list(range(n)[::-1])
    else:
        new_order = list(new_order)

    # transpose
    batch_idcs = list(range(len(batch_shape)))
    new_order_all = new_order + [q for q in range(n) if q not in new_order]
    new_order_all = [q + len(batch_idcs) for q in new_order_all]  # move after batch dimensions
    if len(state.shape) == 1 + len(batch_shape):
        state = state.reshape(batch_shape + [2]*n)
        state = state.transpose(batch_idcs + new_order_all)
        state = state.reshape(batch_shape + [2**n])
    elif len(state.shape) == 2 + len(batch_shape) and state.shape[-2] == state.shape[-1]:
        state = state.reshape(batch_shape + [2,2]*n)
        new_order_all = new_order_all + [i + n for i in new_order_all]
        state = state.transpose(batch_idcs + new_order_all)
        state = state.reshape(batch_shape + [2**n, 2**n])
    else:
        raise ValueError(f"Not a valid shape: {state.shape}")
    return state

def reverse_qubit_order(state, assume_square=True):
    """ So the last will be first, and the first will be last. """
    return transpose_qubit_order(state, -1, assume_square)

def partial_trace(state, retain_qubits, reorder=False, assume_square=True):
    """
    Trace out all qubits not specified in `retain_qubits` and returns the reduced density matrix (or a scalar, if all qubits are traced out).
    If `reorder=True` (default: False), order the output according to `retain_qubits`.
    """
    state = np.asarray(state)
    n = count_qubits(state)

    # add a dummy axis to make it a batch if necessary
    if len(state.shape) == 1:
        state = state[None,:]
        remove_batch = True
        isket = True
    elif len(state.shape) == 2 and state.shape[0] == state.shape[1]:
        if assume_square:
            state = state[None,:,:]
            remove_batch = True
            isket = False
        else:
            remove_batch = False
            isket = True
    else:
        remove_batch = False
        isket = is_ket(state[*[0]*len(state.shape[:-2]),0,:], False)

    if isket:
        # assert_ket(state[*[0]*len(state.shape[:-2]),0,:])
        batch_shape = state.shape[:-1]
    else:
        # assert_dm(state[*[0]*len(state.shape[:-3]),0,:,:], check=1)
        assert len(state.shape) >= 3 and state.shape[-2] == state.shape[-1], f"Invalid state shape {state.shape}"
        batch_shape = state.shape[:-2]
    batch_shape = list(batch_shape)

    # pre-process retain_qubits
    if is_int(retain_qubits):
        retain_qubits = [retain_qubits]
    assert all(0 <= q < n for q in retain_qubits), f"Invalid qubit indices {retain_qubits} for {n}-qubit state"

    # get qubits to trace out
    trace_out = np.array(sorted(set(range(n)) - set(retain_qubits)))

    if isket:
        if len(trace_out) == n:
            return np.linalg.norm(state, axis=-1)  # Tr(|p><p|) = <p|p> -> inner product
        elif len(trace_out) == 0:
            res = state[...,None] * state.conj()[...,None,:]  # outer product
        else:
            state = state.reshape(batch_shape + [2]*n)
            res   = np.zeros(batch_shape + [2]*len(retain_qubits)*2, dtype=state.dtype)
            for idcs in shape_it(batch_shape):
                res[idcs] = np.tensordot(state[idcs], state[idcs].conj(), axes=(trace_out,trace_out))
        state = res.reshape(batch_shape + [2**len(retain_qubits)]*2)
    # if trace out all qubits, just return the normal trace
    elif len(trace_out) == n:
        res = np.trace(state, axis1=-2, axis2=-1).reshape(batch_shape)
        if remove_batch:
            return res[0]
        return res
    else:
        assert state.shape[-2] == state.shape[-1], f"Can't trace a non-square matrix {state.shape}"

        state = state.reshape(batch_shape + [2]*(2*n))
        trace_out = np.array(trace_out) + len(batch_shape)
        for qubit in trace_out:
            state = np.trace(state, axis1=qubit, axis2=qubit+n)
            n -= 1         # one qubit less
            trace_out -= 1 # rename the axes (only "higher" ones are left)
        state = state.reshape(batch_shape + [2**n, 2**n])

    if reorder:
        state = transpose_qubit_order(state, retain_qubits, True)

    if remove_batch:
        return state[0]
    return state

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

    If `kind='fast'`, the state vectors are generated by sampling each element from a normal distribution,
    resulting in sampling from the Haar measure.
    If `kind='ortho'`, generates a set of orthonormal states using QR decomposition. At most $2^n$ orthogonal states can be generated (in the last batch dimension).
    """
    assert is_int(n) and n >= 1, f"n should be an integer >= 1, but was: {n}"
    if not is_iterable(size):
        size = [size]
    assert size == [None] or all(s >= 1 for s in size), f"size should contain only integers >= 1, but was: {size}"
    if kind == 'fast' or size == [None]:
        if size == [None]:
            return normalize(random_vec(2**n, complex=True, kind='normal'))
        kets = random_vec((*size, 2**n), complex=True, kind='normal')
        return normalize(kets, axis=-1)
    elif kind == 'ortho':
        if len(size) > 1:
            res = np.empty([*size, 2**n], dtype=complex)
            for i in shape_it(size[:-1]):
                res[i] = random_ket(n, size[-1], kind='ortho')
            return res
        size = size[0]
        assert size <= 2**n, f"Can't generate more than 2**{n} = {2**n} orthogonal states, but requested was: {size}"
        # inspired by the method to sample Haar-random unitaries
        kets = random_vec((2**n, min(size, 2**n)), complex=True, kind='normal')
        Q, R = np.linalg.qr(kets)
        Rd = np.diag(R)
        L = Rd / np.abs(Rd)
        return L[:,None] * Q.T
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
        return np.outer(state, state.conj())

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

def ket(specification, n=None, renormalize=False, check=1):
    """Convert a string or dictionary of strings and weights to a state vector. The string can be a binary number 
    or a combination of binary numbers and weights. The weights will be normalized to 1."""
    # if a string is given, convert it to a dictionary
    if isinstance(specification, (np.ndarray, list, tuple)):
        n = n or count_qubits(specification) or 1
        specification = np.asarray(specification)
        if len(specification.shape) == 1 and renormalize:
            return normalize(specification)
        if check >= 1:
            assert_ket(specification, n)
        return np.asarray(specification)
    if is_int(specification):
        return ket_from_int(specification, n)
    if type(specification) == str:
        if specification == "random":
            return random_ket(n)
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
            return reduce(np.kron, [s]*n)

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

def unket(state, as_dict=False, prec=5):
    """ Reverse of above.

    `prec` serves as filter for states close to 0, and if `as_dict==False`, it also defines to which precision 
    the values are rounded in the string.

    Example:
    >>> unket(ket('00+01+10+11'))
    '0.5*(00+01+10+11)'
    >>> unket(ket('00+01+10+11'), as_dict=True)
    {'00': 0.5, '01': 0.5, '10': 0.5, '11': 0.5}
    """
    eps = 10**(-prec) if prec is not None else 0
    n = count_qubits(state)
    if as_dict:
        # cast to float if imaginary part is zero
        if np.all(np.abs(state.imag) < 1e-12):
            state = state.real
        return {binstr_from_int(i, n): state[i] for i in range(2**n) if np.abs(state[i]) > eps}

    if prec is not None:
        state = np.round(state, prec)
    # group by weight
    weights = {}
    for i in range(2**n):
        if np.abs(state[i]) > eps:
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

    return np.outer(s1, s2.conj())

def dm(kets, p=None, n=None, renormalize=False, check=3):
    """
    Generate a density matrix from a list of state vectors and probabilities.
    """
    if isinstance(kets, (list, np.ndarray)):
        kets = np.asarray(kets)
        assert len(kets.shape) < 3, f"Invalid shape for state vectors: {kets.shape}"
        if kets.shape[0] == 1 or (len(kets.shape) == 2 and kets.shape[1] == 1):
            kets = kets.ravel()
        if len(kets.shape) == 2:
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
            return random_dm(n, rank='full')
        elif kets == 'random_pure':
            kets = 'random'

    psi = ket(kets, n, renormalize=renormalize, check=check)
    return np.outer(psi, psi.conj())

def as_state(state, renormalize=True, check=2):
    try:
        return ket(state, renormalize=renormalize, check=check)
    except:
        pass
    return dm(state, renormalize=renormalize, check=check)

def ev(obs, state, check=2):
    if check >= 2:
        assert is_hermitian(obs)
    if len(state.shape) == 1:
        assert_ket(state)
        return (state.conj() @ obs @ state).real
    assert_dm(state, check=check)
    return trace_product(obs, state).real

def probs(state):
    """Calculate the probabilities of measuring a state vector in the standard basis."""
    return np.abs(state)**2

def is_ket(psi, n=None, print_errors=True):
    """Check if `ket` is a valid state vector."""
    return is_from_assert(assert_ket, print_errors)(psi, n)

def assert_ket(psi, n=None):
    """ Check if `ket` is a valid state vector. """
    if isinstance(psi, str):
        try:
            psi = ket(psi, check=0)
        except Exception as e:
            assert False, f"Invalid state vector: {psi}\n{e.__class__.__name__}: {e}"
    psi = np.asarray(psi, dtype=complex)
    n = n or count_qubits(psi)
    assert len(psi.shape) == 1 and psi.shape[0] == 2**n, f"Invalid state vector shape: {psi.shape} ≠ {(2**n,)}"
    assert abs(np.linalg.norm(psi) - 1) < 1e-10, f"State vector is not normalized: {np.linalg.norm(psi)}"

def is_dm(rho, n=None, print_errors=True, check=3):
    """Check if matrix `rho` is a density matrix."""
    return is_from_assert(assert_dm, print_errors)(rho, n, check)

def assert_dm(rho, n=None, check=3):
    """ Check if matrix `rho` is a density matrix. """
    if isinstance(rho, (int, str)):
        try:
            rho = dm(rho, renormalize=False, check=0)
        except Exception as e:
            assert False, f"Invalid density matrix: {rho}\n{e.__class__.__name__}: {e}"
    rho = np.asarray(rho, dtype=complex)
    assert len(rho.shape) == 2 and rho.shape[0] == rho.shape[1], f"Invalid density matrix shape: {rho.shape}"
    n = n or count_qubits(rho)
    assert rho.shape[0] == 2**n, f"Invalid density matrix shape: {rho.shape} should be a power of 2"
    assert abs(np.trace(rho) - 1) < 1e-10, f"Density matrix is not normalized: {np.trace(rho)}"
    assert is_psd(rho, check=check), "Density matrix is not positive semi-definite"

def is_state(state, check=3):
    try:
        assert_ket(state)
        return True
    except:
        pass
    try:
        assert_dm(state, check=check)
        return True
    except:
        return False

def is_pure_dm(rho, check=3, tol=1e-12):
    """ Check if matrix `rho` is a pure density matrix. """
    if not is_dm(rho, check=check):
        return False
    # return np.linalg.matrix_rank(rho) == 1
    if check >= 1:
        return abs(trace_product(rho, rho) - 1) < tol
    return True

def is_eigenstate(psi, H, tol=1e-10):
    psi = normalize(psi)
    psi2 = normalize(H @ psi)
    return abs(abs(psi2.conj() @ psi) - 1) < tol

def gibbs(H, beta=1, check=2):
    """Calculate the Gibbs state of a Hamiltonian `H` at inverse temperature `beta`."""
    H = np.asarray(H)
    if check >= 2:
        assert is_hermitian(H), "Hamiltonian must be Hermitian"
    assert beta >= 0, f"Inverse temperature must be positive, but was {beta}"
    E, U = eigh(H)
    E = softmax(E, -beta)
    return U @ (E[:,None] * U.conj().T)

def count_qubits(obj):
    def asint(x):
        n = int(log2(x))
        assert x == 2**n, f"Dimension must be a power of 2, but was {x}"
        return n

    if isinstance(obj, str) or (hasattr(obj, 'dtype') and obj.dtype.kind == 'U'):  # after conversion to numpy array
        if hasattr(obj, 'item'):
            obj = obj.item()
        import re
        obj = obj.replace(' ', '')  # remove spaces
        if "+" in obj or "-" in obj:
            if 'X' in obj or 'Y' in obj or 'Z' in obj or 'I' in obj:
                # obj = parse_hamiltonian(obj)
                return len(re.search('[XYZI]+', obj)[0])
            else:
                # obj = ket(obj)
                return len(re.search('[01]+(?![.*01])', obj)[0])
        else:
            # obj = parse_unitary(obj)
            obj = obj.split('@')[0]
            return len(re.search('\\S+', obj)[0])
    if hasattr(obj, 'n'):
        return obj.n
    if hasattr(obj, 'shape'):
        return asint(obj.shape[-1])
    if hasattr(obj, '__len__'):
        return asint(len(obj))
    if is_int(obj):
        return asint(obj)
    if hasattr(obj, 'num_qubits'):
        return obj.num_qubits
    if hasattr(obj, 'qubits'):
        return len(obj.qubits)
    raise ValueError(f'Unkown object: {obj}')

## TODO: def random_stabilizer_state(n): + convert n to full quantum state
## TODO: random_tensor_network_state + convert tensor network state -> full quantum state