from math import log2
import numpy as np

from ..utils import is_int

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

def transpose_qubit_order(state, new_order, assume_square=True):
    state = np.asarray(state)
    n = count_qubits(state)

    # parse new_order
    if isinstance(new_order, int) and new_order == -1:
        new_order = list(range(n)[::-1])
    else:
        new_order = list(new_order)
    assert all(0 <= q < n for q in new_order), f"Invalid qubit order: {new_order}"
    assert len(set(new_order)) == len(new_order), f"Invalid qubit order: {new_order}"

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

    # transpose
    batch_idcs = list(range(len(batch_shape)))
    new_order_all = new_order + [q for q in range(n) if q not in new_order]
    new_order_all = [q + len(batch_idcs) for q in new_order_all]  # move after batch dimensions
    if len(state.shape) == 1 + len(batch_shape):  # vector
        state = state.reshape(batch_shape + [2]*n)
        state = state.transpose(batch_idcs + new_order_all)
        state = state.reshape(batch_shape + [2**n])
    elif len(state.shape) == 2 + len(batch_shape) and state.shape[-2] == state.shape[-1]:  # matrix
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