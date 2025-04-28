from math import log2
import numpy as np

from ..mathlib import is_square, outer
from ..utils import is_int, shape_it

def verify_subsystem(subsystem, n):
    if is_int(subsystem):
        subsystem = [subsystem]
    subsystem = list(subsystem)
    assert all(0 <= q < n for q in subsystem), f"Subsystem is has invalid qubits: {subsystem}"
    return subsystem

def count_qubits(obj):
    def asint(x):
        n = int(log2(x))
        assert x == 2**n, f"Dimension must be a power of 2, but was {x}"
        return n

    if is_int(obj):
        return asint(obj)
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
    if hasattr(obj, 'shape'):
        return asint(obj.shape[-1])  # input space
    if hasattr(obj, 'n'):
        return obj.n
    if hasattr(obj, '__len__'):
        return asint(len(obj))
    if hasattr(obj, 'num_qubits'):
        return obj.num_qubits
    if hasattr(obj, 'qubits'):
        return len(obj.qubits)
    raise ValueError(f'Unkown object: {obj}')

def transpose_qubit_order(state, new_order, reshape=False, batch_shape=()):
    state = np.asarray(state)
    n = count_qubits(state)

    # parse new_order
    if isinstance(new_order, int) and new_order == -1:
        new_order = range(n)[::-1]
    new_order = verify_subsystem(new_order, n)
    q = len(new_order)
    new_order_all = new_order + [q for q in range(n) if q not in new_order]

    # parse batch shape
    batch_idcs, batch_shape = [], list(batch_shape)
    if batch_shape:
        batch_ndim = len(batch_shape)
        is_ket = state.ndim - batch_ndim == 1
        batch_idcs = list(range(batch_ndim))
        new_order_all = [q + batch_ndim for q in new_order_all]  # move after batch dimensions
    else:
        is_ket = state.ndim == 1

    # transpose
    if is_ket:
        state = state.reshape(batch_shape + [2]*n)
        state = state.transpose(batch_idcs + new_order_all)
        if reshape:
            state = state.reshape(batch_shape + [2**q, 2**(n-q)])
        else:
            state = state.reshape(batch_shape + [2**n])
    elif is_square(state):
        state = state.reshape(batch_shape + [2,2]*n)
        new_order_all = new_order_all + [i + n for i in new_order_all]
        state = state.transpose(batch_idcs + new_order_all)
        if reshape:
            state = state.reshape(batch_shape + [2**q, 2**(n-q), 2**q, 2**(n-q)])
        else:
            state = state.reshape(batch_shape + [2**n, 2**n])
    # TODO: allow non-square operators
    else:
        raise ValueError(f"Not a valid shape: {state.shape}")
    return state

def reverse_qubit_order(state, batch_shape=()):
    """ So the last will be first, and the first will be last. """
    return transpose_qubit_order(state, -1, reshape=False, batch_shape=batch_shape)

def partial_trace(state, retain_qubits, reorder=False, assume_ket=False):
    """
    Trace out all qubits not specified in `retain_qubits` and returns the reduced density matrix (or a scalar, if all qubits are traced out).
    If `reorder=True` (default: False), order the output according to `retain_qubits`.
    """
    state = np.asarray(state)
    n = count_qubits(state)

    # add a dummy axis to make it a batch if necessary
    if state.ndim == 1 or assume_ket or not is_square(state):
        isket = True
        batch_shape = state.shape[:-1]
        # psi = state[*[0]*len(state.shape[:-2]),0,:]
        # isket = len(psi) == 2**n and abs(np.linalg.norm(psi) - 1) < 1e-10
        # assert isket
    else:
        isket = False
        batch_shape = state.shape[:-2]
    batch_shape = list(batch_shape)

    # pre-process retain_qubits
    retain_qubits = verify_subsystem(retain_qubits, n)

    # get qubits to trace out
    trace_out = np.array(sorted(set(range(n)) - set(retain_qubits)))

    if isket:
        if len(trace_out) == n:
            return np.linalg.norm(state, axis=-1)  # Tr(|p><p|) = <p|p> -> inner product
        elif len(trace_out) == 0:
            res = outer(state)
        else:
            state = state.reshape(batch_shape + [2]*n)
            res   = np.zeros(batch_shape + [2]*len(retain_qubits)*2, dtype=state.dtype)
            for idcs in shape_it(batch_shape):
                res[idcs] = np.tensordot(state[idcs], state[idcs].conj(), axes=(trace_out,trace_out))
        state = res.reshape(batch_shape + [2**len(retain_qubits)]*2)
    # if trace out all qubits, just return the normal trace
    elif len(trace_out) == n:
        state = np.trace(state, axis1=-2, axis2=-1).reshape(batch_shape)
        reorder = False
    else:
        assert is_square(state), f"Can't trace a non-square matrix {state.shape}"

        state = state.reshape(batch_shape + [2]*(2*n))
        trace_out = np.array(trace_out) + len(batch_shape)
        for qubit in trace_out:
            state = np.trace(state, axis1=qubit, axis2=qubit+n)
            n -= 1         # one qubit less
            trace_out -= 1 # rename the axes (only "higher" ones are left)
        state = state.reshape(batch_shape + [2**n, 2**n])

    if reorder:
        state = transpose_qubit_order(state, np.argsort(retain_qubits), reshape=False, batch_shape=batch_shape)

    return state