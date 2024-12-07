import psutil
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from math import log2

from ..utils import is_int, duh, is_from_assert
from ..mathlib import normalize, binstr_from_int, is_hermitian, softmax, is_psd, random_vec
from ..plot import colorize_complex
from ..prob import random_p

def transpose_qubit_order(state, new_order):
    state = np.asarray(state)
    n = count_qubits(state)
    if new_order == -1:
        new_order = list(range(n)[::-1])
    else:
        new_order = list(new_order)

    new_order_all = new_order + [q for q in range(n) if q not in new_order]
    if len(state.shape) == 1:
        state = state.reshape([2]*n)
        state = state.transpose(new_order_all)
        state = state.reshape(-1)
    elif len(state.shape) == 2 and state.shape[0] == state.shape[1]:
        state = state.reshape([2,2]*n)
        new_order_all = new_order_all + [i + n for i in new_order_all]
        state = state.transpose(new_order_all)
        state = state.reshape([2**n, 2**n])
    else:
        raise ValueError(f"Not a valid shape: {state.shape}")
    return state

def reverse_qubit_order(state):
    """So the last will be first, and the first will be last."""
    return transpose_qubit_order(state, -1)

def partial_trace(state, retain_qubits, reorder=False):
    """
    Trace out all qubits not specified in `retain_qubits` and returns the reduced density matrix (or a scalar, if all qubits are traced out).
    If `reorder=True` (default: False), order the output according to `retain_qubits`.
    """
    state = np.asarray(state)
    n = count_qubits(state)

    # pre-process retain_qubits
    if is_int(retain_qubits):
        retain_qubits = [retain_qubits]
    assert all(0 <= q < n for q in retain_qubits), f"Invalid qubit indices {retain_qubits} for {n}-qubit state"

    # get qubits to trace out
    trace_out = np.array(sorted(set(range(n)) - set(retain_qubits)))

    # if rho is a state vector
    if len(state.shape) == 1:
        if len(trace_out) == n:
            # Tr(|p><p|) = <p|p>
            return state @ state.conj()
        elif len(trace_out) == 0:
            if reorder:
                state = transpose_qubit_order(state, retain_qubits)
            return np.outer(state, state.conj())
        state = state.reshape([2]*n)
        state = np.tensordot(state, state.conj(), axes=(trace_out,trace_out))
    # if trace out all qubits, just return the normal trace
    elif len(trace_out) == n:
        return np.trace(state).reshape(1,1)
    else:
        assert state.shape[0] == state.shape[1], f"Can't trace a non-square matrix {state.shape}"

        state = state.reshape([2]*(2*n))
        for qubit in trace_out:
            state = np.trace(state, axis1=qubit, axis2=qubit+n)
            n -= 1         # one qubit less
            trace_out -= 1 # rename the axes (only "higher" ones are left)

    if reorder:
        # transpose the axes of the remaining qubits to the desired order
        new_order = np.argsort(list(retain_qubits))
        state = state.transpose(new_order.tolist() + (new_order+len(new_order)).tolist())

    return state.reshape([2**len(retain_qubits)]*2)

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
        if i not in retain_qubits:
            state = np.sum(state, axis=cur)
            probs = np.sum(probs, axis=cur)
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

    state = ket(state)

    # trace out unwanted qubits
    if showqubits is None:
        n = count_qubits(state)
        showqubits = range(n)

    if showrho:
        memory_requirement = (len(state))**2 * 16
        #print(memory_requirement / 1024**2, "MB") # rho.nbytes
        if memory_requirement > psutil.virtual_memory().available:
            raise ValueError(f"Too much memory required ({duh(memory_requirement)}) to calulate the density matrix!")
        rho = partial_trace(rho, retain_qubits=showqubits, reorder=True)

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

def random_ket(n=1, kind='haar'):
    """ Generate a random state vector ($2^{n+1}-1$ degrees of freedom). """
    return random_kets(n, 1, kind=kind)[0]

def random_kets(n, size=1, kind='haar'):
    assert is_int(n) and n >= 1, f"n should be an integer >= 1, but was: {n}"
    assert is_int(size) and size >= 1, f"size should be an integer >= 1, but was: {size}"
    if kind == 'fast':
        kets = random_vec((size, 2**n), complex=True, kind='normal')
        return normalize(kets, axis=1)
    elif kind == 'haar':
        if size > 2**n:
            res = [random_kets(n, 2**n) for _ in range(size // 2**n)]
            if size % 2**n != 0:
                res += [random_kets(n, size % 2**n, 'haar')]
            return np.concatenate(res, axis=0)
        kets = random_vec((2**n, min(size, 2**n)), complex=True, kind='normal')
        Q, R = np.linalg.qr(kets)
        Rd = np.diag(R)
        L = Rd / np.abs(Rd)
        return L[:,None] * Q.T
    else:
        raise ValueError(f"Unknown kind: {kind}")

def random_dm(n=1, rank='full', kind='haar'):
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
        state = random_ket(n, kind=kind)
        return np.outer(state, state.conj())

    probs = random_p(rank, kind='uniform')
    kets = random_kets(n, rank, kind=kind)
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

def ket(specification, n=None, check=1):
    """Convert a string or dictionary of strings and weights to a state vector. The string can be a binary number 
    or a combination of binary numbers and weights. The weights will be normalized to 1."""
    # if a string is given, convert it to a dictionary
    if isinstance(specification, (np.ndarray, list, tuple)):
        n = n or count_qubits(specification) or 1
        if hasattr(specification, 'shape'):
            assert specification.shape == (2**n,), f"State vector has wrong shape for {n} qubits: {specification.shape} ≠ {(2**n,)}!"
        else:
            assert len(specification) == 2**n, f"State vector has wrong size for {n} qubits: {len(specification)} ≠ {2**n}!"
        if check >= 1:
            return normalize(specification)
        return np.asarray(specification)
    if is_int(specification):
        return ket_from_int(specification, n)
    if type(specification) == str:
        if specification == "random":
            return random_ket(n)
        # handle some special cases: |+>, |->, |i>, |-i>
        if specification == "+":
            return normalize(np.array([1,1], dtype=complex))
        elif specification == "-":
            return normalize(np.array([1,-1], dtype=complex))
        elif specification == "i":
            return normalize(np.array([1,1j], dtype=complex))
        elif specification == "-i":
            return normalize(np.array([1,-1j], dtype=complex))

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
    """ Reverse of above. The output is always guaranteed to be normalized.

    `prec` serves as filter for states close to 0, and if `as_dict==False`, it also defines to which precision 
    the values are rounded in the string.

    Example:
    >>> unket(ket('00+01+10+11'))
    '0.5*(00+01+10+11)'
    >>> unket(ket('00+01+10+11'), as_dict=True)
    {'00': 0.5, '01': 0.5, '10': 0.5, '11': 0.5}
    """
    eps = 10**(-prec) if prec is not None else 0
    state = normalize(state)
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

def dm(specification1, specification2=None, n=None, check=2, obs=None):
    """
    Convenience function to generate or verify a density matrix. Also allows to generate other operators, e.g. `dm(0, 1)`.
    Check level only relevant if a matrix or observable is given.
    """
    # If it's already a matrix, ensure it's a density matrix and return it
    if isinstance(specification1, (list, np.ndarray)):
        specification1 = np.asarray(specification1)
        if len(specification1.shape) > 1:
            n = n or count_qubits(specification1) or 1
            assert specification1.shape == (2**n, 2**n), f"Matrix has wrong shape for {n} qubits: {specification1.shape} ≠ {(2**n, 2**n)}"
            if check >= 1:
                sp1_trace = np.trace(specification1)
                # trace normalize it only if it's not already
                if not abs(sp1_trace - 1) < 1e-8:
                    specification1 = specification1 / sp1_trace
            assert_dm(specification1, check=check)
            return specification1
    elif specification2 is None:
        if specification1 == 'random' or specification1 == 'random_mixed' or specification1 == 'random_dm':
            return random_dm(n, rank='full')
        elif specification1 == 'random_pure':
            specification1 = 'random'

    s1 = ket(specification1, n, check=check)
    if obs is not None:
        if check >= 2:
            assert is_hermitian(obs), "The given observable is not Hermitian"
        U = eigh(obs)[1]
        s1 = U @ s1
    if specification2 is None:
        s2 = s1
    else:
        s2 = ket(specification2, n or count_qubits(s1), check=check)
        if obs is not None:
            s2 = U @ s2

    return np.outer(s1, s2.conj())

def as_state(state, check=2):
    try:
        return ket(state, check=check)
    except:
        pass
    return dm(state, check=check)

def ev(obs, state, check=2):
    if check >= 2:
        assert is_hermitian(obs)
    if len(state.shape) == 1:
        assert_ket(state)
        return (state.conj() @ obs @ state).real
    assert_dm(state, check=check)
    return np.trace(obs @ state).real

def probs(state):
    """Calculate the probabilities of measuring a state vector in the standard basis."""
    return np.abs(state)**2

def is_ket(psi, print_errors=True):
    """Check if `ket` is a valid state vector."""
    return is_from_assert(assert_ket, print_errors)(psi)

def assert_ket(psi):
    """ Check if `ket` is a valid state vector. """
    if isinstance(psi, str):
        try:
            psi = ket(psi, check=0)
        except Exception as e:
            assert False, f"Invalid state vector: {psi}\n{e.__class__.__name__}: {e}"
    psi = np.asarray(psi, dtype=complex)
    n = count_qubits(psi)
    assert len(psi.shape) == 1 and psi.shape[0] == 2**n, f"Invalid state vector shape: {psi.shape} ≠ {(2**n,)}"
    assert abs(np.linalg.norm(psi) - 1) < 1e-10, f"State vector is not normalized: {np.linalg.norm(psi)}"

def is_dm(rho, print_errors=True, check=3):
    """Check if matrix `rho` is a density matrix."""
    return is_from_assert(assert_dm, print_errors)(rho, check)

def assert_dm(rho, check=3):
    """ Check if matrix `rho` is a density matrix. """
    if isinstance(rho, (int, str)):
        try:
            rho = dm(rho, check=0)
        except Exception as e:
            assert False, f"Invalid density matrix: {rho}\n{e.__class__.__name__}: {e}"
    rho = np.asarray(rho, dtype=complex)
    assert len(rho.shape) == 2 and rho.shape[0] == rho.shape[1], f"Invalid density matrix shape: {rho.shape}"
    n = count_qubits(rho)
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
    if check >= 2:
        return abs(np.trace(rho @ rho) - 1) < tol

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
    if hasattr(obj, '__len__'):
        n = int(log2(len(obj)))
        # assert len(obj) == 2**n, f"Dimension must be a power of 2, but was {len(obj)}"
        return n
    if hasattr(obj, 'num_qubits'):
        return obj.num_qubits
    if hasattr(obj, 'qubits'):
        return len(obj.qubits)
    raise ValueError(f'Unkown object: {obj}')

## TODO: def random_stabilizer_state(n): + convert n to full quantum state
## TODO: random_tensor_network_state + convert tensor network state -> full quantum state