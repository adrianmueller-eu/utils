import psutil, warnings
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh, eigvalsh
import itertools
from functools import reduce

from ..mathlib import normalize, sequence, softmax, is_hermitian, pauli_basis, allclose0, is_unitary, binstr_from_float
from ..utils import duh
from .state import random_ket, unket

def ground_state_exact(hamiltonian):
    """ Calculate the ground state using exact diagonalization. """
    if isinstance(hamiltonian, str):
        H = parse_hamiltonian(hamiltonian, sparse=True)
    # elif isinstance(hamiltonian, nk.operator.DiscreteOperator):
    #     H = hamiltonian.to_sparse()
    else:
        H = hamiltonian

    if sp.issparse(H):
        evals, evecs = sp.linalg.eigsh(H, k=1, which='SA')
    else:
        evals, evecs = eigh(H) # way faster, but only dense matrices
    ground_state_energy = evals[0]
    ground_state = evecs[:,0]
    return ground_state_energy, ground_state

def get_E0(H):
    return ground_state_exact(H)[0]

def ground_state_ITE(H, tau=5, eps=1e-6, check=2):  # eps=1e-6 gives almost perfect precision in the energy
    """ Calculate the ground state using the Imaginary Time-Evolution (ITE) scheme.
    Since its vanilla form uses diagonalization (to calculate the matrix exponential), it can't be more efficient than diagonalization itself. """
    def evolve(i, psi):
        psi = U @ psi
        return normalize(psi)

    # U = matexp(-tau*H)
    if check >= 2:
        assert is_hermitian(H)
    D, V = eigh(H)  # this requires diagonalization of H
    U = V @ (softmax(D, -tau)[:,None] * V.conj().T)
    n = int(np.log2(H.shape[0]))
    ground_state = sequence(evolve, start_value=random_ket(n), eps=eps)
    ground_state_energy = (ground_state.conj().T @ H @ ground_state).real
    return ground_state_energy, ground_state

matmap_np, matmap_sp = None, None

def parse_hamiltonian(hamiltonian, sparse=False, scaling=1, buffer=None, max_buffer_n=0, dtype=complex, check=2):
    """Parse a string representation of a Hamiltonian into a matrix representation. The result is guaranteed to be Hermitian.
    The string `hamiltonian` must follow the following syntax (see examples below)
        ```
        <hamiltonian> = <term> | <term> + <hamiltonian>
        <term>        = <pauli> | (<hamiltonian>) | <weight>*<pauli> | <weight>*(<hamiltonian>) | <weight>
        <pauli>       = [IXYZ]+
        ```
    where `<weight>` is a string representing a finite number that can be parsed by `float()`. If `<weight>` is not followed by `*`, it is assumed to be the respective multiple of the identity.

    Parameters:
        hamiltonian (str): The Hamiltonian to parse.
        sparse (bool): Whether to use sparse matrices (csr_matrix) or dense matrices (numpy.array).
        scaling (float): A constant factor to scale the Hamiltonian by.
        buffer (dict): A dictionary to store calculated chunks in. If `None`, it defaults to the global `matmap_np` (or `matmap_sp` if `sparse == True`). Give `buffer={}` and leave `max_buffer_n == 0` (default) to disable the buffer.
        max_buffer_n (int): The maximum length (number of qubits) for new chunks to store in the buffer (default: 0). If `0`, no new chunks will be stored in the buffer.
        dtype (type): The data type of the matrix elements.
        check (int): Checking whether the resulting matrix is Hermitian requires check level 2.

    Returns:
        numpy.ndarray | scipy.sparse.csr_matrix: The matrix representation of the Hamiltonian.

    Example:
    >>> parse_hamiltonian('0.5*(XX + YY + ZZ + II)') # SWAP
    array([[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
           [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
           [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
           [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]])
    >>> parse_hamiltonian('X - 2*Y + 1')
    array([[1.+0.j, 1.+2.j],
           [1.-2.j, 1.+0.j]])
    >>> parse_hamiltonian('-(XX + YY + .5*ZZ) + 1.5')
    array([[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
           [ 0.+0.j  2.+0.j -2.+0.j  0.+0.j]
           [ 0.+0.j -2.+0.j  2.+0.j  0.+0.j]
           [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]])
    >>> parse_hamiltonian('0.5*(II + ZI - ZX + IX)') # CNOT

    """
    kron = sp.kron if sparse else np.kron

    # Initialize the matrix map
    global matmap_np, matmap_sp
    if matmap_np is None or matmap_sp is None or matmap_np["I"].dtype != dtype:
        # numpy versions
        matmap_np = {
            "H": np.array([[1, 1], [1, -1]], dtype=dtype) / np.sqrt(2),
            "X": np.array([[0, 1], [1, 0]], dtype=dtype),
            "Z": np.array([[1, 0], [0, -1]], dtype=dtype),
            "I": np.array([[1, 0], [0, 1]], dtype=dtype),
        }
        # composites
        matmap_np.update({
            "ZZ": np.kron(matmap_np['Z'], matmap_np['Z']),
            "IX": np.kron(matmap_np['I'], matmap_np['X']),
            "XI": np.kron(matmap_np['X'], matmap_np['I']),
            "YY": np.array([[ 0,  0,  0, -1],  # to avoid complex numbers
                            [ 0,  0,  1,  0],
                            [ 0,  1,  0,  0],
                            [-1,  0,  0,  0]], dtype=dtype)
        })
        for i in range(2, 11):
            matmap_np["I"*i] = np.eye(2**i, dtype=dtype)
        # add 'Y' only if dtype supports imaginary numbers
        if np.issubdtype(dtype, np.complexfloating):
            matmap_np["Y"] = np.array([[0, -1j], [1j, 0]], dtype=dtype)

        # sparse versions
        matmap_sp = {k: sp.csr_array(v) for k, v in matmap_np.items()}

    if not np.issubdtype(dtype, np.complexfloating) and "Y" in hamiltonian:
        raise ValueError(f"The Pauli matrix Y is not supported for dtype {dtype.__name__}.")

    matmap = matmap_sp if sparse else matmap_np

    # only use buffer if pre-computed chunks are available or if new chunks are allowed to be stored
    use_buffer = buffer is None or len(buffer) > 0 or max_buffer_n > 0
    if use_buffer and buffer is None:
        buffer = matmap

    def calculate_chunk_matrix(chunk, sparse=False, scaling=1):
        # if scaling != 1:  # only relevant for int dtype, to prevent changing dtype when multiplying
            # scaling = np.array(scaling, dtype=dtype)
        if use_buffer:
            if chunk in buffer:
                return buffer[chunk] if scaling == 1 else scaling * buffer[chunk]
            if len(chunk) == 1:
                return matmap[chunk[0]] if scaling == 1 else scaling * matmap[chunk[0]]
            # Check if a part of the chunk has already been calculated
            for i in range(len(chunk)-1, 1, -1):
                for j in range(len(chunk)-i+1):
                    subchunk = chunk[j:j+i]
                    if subchunk in buffer:
                        # If so, calculate the rest of the chunk recursively
                        parts = [chunk[:j], subchunk, chunk[j+i:]]
                        # remove empty chunks
                        parts = [c for c in parts if c != ""]
                        # See where to apply the scaling
                        shortest = min(parts, key=len)
                        # Calculate each part recursively
                        for i, c in enumerate(parts):
                            if c == subchunk:
                                if c == shortest:
                                    parts[i] = scaling * buffer[c]
                                    shortest = ""
                                else:
                                    parts[i] = buffer[c]
                            else:
                                if c == shortest:
                                    parts[i] = calculate_chunk_matrix(c, sparse=sparse, scaling=scaling)
                                    shortest = ""
                                else:
                                    parts[i] = calculate_chunk_matrix(c, sparse=sparse, scaling=1)
                        return reduce(kron, parts)

        # Calculate the chunk matrix gate by gate
        if use_buffer and len(chunk) <= max_buffer_n:
            gates = [matmap[gate] for gate in chunk]
            chunk_matrix = reduce(kron, gates)
            buffer[chunk] = chunk_matrix
            if scaling != 1:
                chunk_matrix = scaling * chunk_matrix
        else:
            gates = [scaling * matmap[chunk[0]]] + [matmap[gate] for gate in chunk[1:]]
            chunk_matrix = reduce(kron, gates)

        return chunk_matrix

    # Remove whitespace
    hamiltonian = hamiltonian.replace(" ", "")
    # replace - with +-, except before e
    hamiltonian = hamiltonian \
                    .replace("-", "+-") \
                    .replace("e+-", "e-") \
                    .replace("(+-", "(-")

    # print("ph: Pre-processed Hamiltonian:", hamiltonian)

    # Find parts in parentheses
    part = ""
    parts = []
    depth = 0
    current_part_weight = ""
    for i, c in enumerate(hamiltonian):
        if c == "(":
            if depth == 0:
                # for top-level parts search backwards for the weight
                weight = ""
                for j in range(i-1, -1, -1):
                    if hamiltonian[j] in ["("]:
                        break
                    weight += hamiltonian[j]
                    if hamiltonian[j] in ["+", "-"]:
                        break
                weight = weight[::-1]
                if weight != "":
                    current_part_weight = weight
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                part += c
                parts.append((current_part_weight, part))
                part = ""
                current_part_weight = ""
        if depth > 0: 
            part += c

    # print("Parts found:", parts)

    # Replace parts in parentheses with a placeholder
    for i, (weight, part) in enumerate(parts):
        hamiltonian = hamiltonian.replace(weight+part, f"+part{i}", 1)
        # remove * at the end of the weight
        if weight != "" and weight[-1] == "*":
            weight = weight[:-1]
        if weight in ["", "+", "-"]:
            weight += "1"
        # Calculate the part recursively
        part = part[1:-1] # remove parentheses
        parts[i] = parse_hamiltonian(part, sparse=sparse, scaling=float(weight), buffer=buffer, max_buffer_n=max_buffer_n, dtype=dtype)

    # print("Parts replaced:", parts)

    # Parse the rest of the Hamiltonian
    chunks = hamiltonian.split("+")
    # Remove empty chunks
    chunks = [c for c in chunks if c != ""]
    # If parts are present, use them to determine the number of qubits
    if parts:
        n = int(np.log2(parts[0].shape[0]))
    else: # Use chunks to determine the number of qubits
        n = 0
        for c in chunks:
            if c[0] in ["-", "+"]:
                c = c[1:]
            if "*" in c:
                c = c.split("*")[1]
            if c.startswith("part"):
                continue
            try:
                float(c)
                continue
            except ValueError:
                n = len(c)
                break
        if n == 0:
            warnings.warn("Hamiltonian is a scalar!", stacklevel=2)

    if not sparse and n > 10:
        # check if we would blow up the memory
        mem_required = 2**(2*n) * np.array(1, dtype=dtype).nbytes
        mem_available = psutil.virtual_memory().available
        if mem_required > mem_available:
            raise MemoryError(f"This would blow up you memory ({duh(mem_required)} required)! Try using `sparse=True`.")

    if sparse:
        H = sp.csr_array((2**n, 2**n), dtype=dtype)
    else:
        if n > 10:
            warnings.warn(f"Using a dense matrix for a {n}-qubit Hamiltonian is not recommended. Use sparse=True.", stacklevel=2)
        H = np.zeros((2**n, 2**n), dtype=dtype)

    for chunk in chunks:
        # print("Processing chunk:", chunk)
        if chunk == "":
            continue
        chunk_matrix = None
        # Parse the weight of the chunk
        if chunk.startswith("part"):
            weight = 1  # parts already include their weight
            chunk_matrix = parts[int(chunk.split("part")[1])]
        elif "*" in chunk:
            weight = float(chunk.split("*")[0])
            chunk = chunk.split("*")[1]
        elif len(chunk) == n+1 and chunk[0] in ["-", "+"] and n >= 1 and chunk[1] in matmap:
            weight = float(chunk[0] + "1")
            chunk = chunk[1:]
        elif (chunk[0] in ["-", "+", "."] or chunk[0].isdigit()) and all([c not in matmap for c in chunk[1:]]):
            if len(chunk) == 1 and chunk[0] in ["-", "."]:
                chunk = 0
            weight = complex(chunk)
            if np.iscomplex(weight):
                raise ValueError("Complex scalars would make the Hamiltonian non-Hermitian!")
            weight = weight.real
            # weight = np.array(weight, dtype=dtype)  # only relevant for int dtype
            if sparse:
                chunk_matrix = sp.eye(2**n, dtype=dtype)
            else:
                chunk_matrix = np.eye(2**n, dtype=dtype)
        elif len(chunk) != n:
            raise ValueError(f"Gate count must be {n} but was {len(chunk)} for chunk \"{chunk}\"")
        else:
            weight = 1

        if chunk_matrix is None:
            chunk_matrix = calculate_chunk_matrix(chunk, sparse=sparse, scaling=scaling * weight)
        elif scaling * weight != 1:
            chunk_matrix = scaling * weight * chunk_matrix

        # Add the chunk to the Hamiltonian
        # print("Adding chunk", weight, chunk, "for hamiltonian", scaling, hamiltonian)
        # print(type(H), H.dtype, type(chunk_matrix), chunk_matrix.dtype)
        if len(chunks) == 1:
            H = chunk_matrix
        else:
            H += chunk_matrix

    if check >= 2:
        if sparse:
            assert allclose0(H.data - H.conj().T.data), f"The given Hamiltonian {hamiltonian} is not Hermitian: {H.data}"
        else:
            assert is_hermitian(H), f"The given Hamiltonian {hamiltonian} is not Hermitian: {H}"

    return H

def random_ham(n_qubits, n_terms, offset=0, coeffs=True, scaling=True):
    """ Draw `n_terms` basis elements out of the basis. If `scaling=True`, the coefficients are normalized to 1. If `coeffs=False`, the coefficients (and scaling) are omitted."""
    # generate a list of random terms
    basis = pauli_basis(n_qubits, kind='str')[1:]  # exclude the identity
    assert n_terms <= len(basis), f"Can't draw {n_terms} terms from a basis of size {len(basis)}!"
    terms = np.random.choice(basis, size=n_terms, replace=False)
    if not coeffs:
        return ' + '.join(terms)
    # generate a random coefficient for each term
    coeffs = np.random.random(n_terms)
    if scaling:
        coeffs = normalize(coeffs, p=1)
    # generate the Hamiltonian
    H_str = ' + '.join([f'{coeffs[i]}*{term}' for i, term in enumerate(terms)])
    if offset != 0:
        H_str += ' + ' + str(offset)
    return H_str

def ising(n_qubits, J=(-1,1), h=(-1,1), g=(-1,1), offset=0, kind='1d', circular=False, as_dict=False, prec=42):
    """
    Generates an Ising model with (optional) longitudinal and (optional) transverse couplings.

    Parameters
    ----------
    n_qubits : int or tuple
        Number of qubits, at least 2. For `kind='2d'` or `kind='3d'`, give a tuple of 2 or 3 integers, respectively.
    J : float, array, or dict
        Coupling strength.
        If a scalar, all couplings are set to this value.
        If a 2-element vector but `n_qubit > 2` (or tuple), all couplings are set to a random value in this range.
        If a matrix, this matrix is used as the coupling matrix.
        For `kind='pairwise'`, `kind='2d'`, or `kind='3d'`, `J` is read as an incidence matrix, where the rows and columns correspond to the qubits and the values are the coupling strengths. Only the upper triangular part of the matrix is used.
        For `kind='all'`, specify a dictionary with keys being tuples of qubit indices and values being the corresponding coupling strength.
    h : float or array, optional
        Longitudinal field strength.
        If a scalar, all fields are set to this value.
        If a 2-element vector, all fields are set to a random value in this range.
        If a vector of size `n_qubits`, its elements specify the individual strengths of the longitudinal field.
    g : float or array, optional
        Transverse field strength.
        If a scalar, all couplings are set to this value.
        If a 2-element vector, all couplings are set to a random value in this range.
        If a vector of size `n_qubits`, its elements specify the individual strengths of the transverse field.
    offset : float, optional
        Offset of the Hamiltonian.
    kind : {'1d', '2d', '3d', 'pairwise', 'all'}, optional
        Whether the couplings are along a string (`1d`), on a 2d-lattice (`2d`), 3d-lattice (`3d`), fully connected graph (`pairwise`), or specify the desired multi-particle interactions.
    circular : bool, optional
        Whether the couplings are circular (i.e. the outermost qubits are coupled to each other). Only applies to `kind='1d'`, `kind='2d'`, and `kind='3d'`.

    Returns
    -------
    H : str
        The Hamiltonian as a string, which can be parsed by ph.
    """
    # TODO: interactions on X and Y, and local fields on Y
    # if n_qubits is not scalar or tuple, try ising_graph
    if not (np.isscalar(n_qubits) or isinstance(n_qubits, tuple)):
        return ising_graph(n_qubits, J=J, h=h, g=g, offset=offset)
    # generate the coupling shape
    n_total_qubits = np.prod(n_qubits)
    if n_total_qubits < 1:
        raise ValueError(f"Number of qubits must be positive, but is {n_total_qubits}")
    assert n_total_qubits - int(n_total_qubits) == 0, "n_qubits must be an integer or a tuple of integers"
    if kind == '1d':
        assert np.isscalar(n_qubits) or len(n_qubits) == 1, f"For kind={kind}, n_qubits must be an integer or tuple of length 1, but is {n_qubits}"
        # convert to int if tuple (has attr __len__)
        if hasattr(n_qubits, '__len__'):
            n_qubits = n_qubits[0]
        couplings = (n_qubits if circular and n_qubits > 2 else n_qubits-1,)
    elif kind == '2d':
        if np.isscalar(n_qubits) or len(n_qubits) != 2:
            raise ValueError(f"For kind={kind}, n_qubits must be a tuple of length 2, but is {n_qubits} ({type(n_qubits)})")
        couplings = (n_total_qubits, n_total_qubits)
    elif kind == '3d':
        if np.isscalar(n_qubits) or len(n_qubits) != 3:
            raise ValueError(f"For kind={kind}, n_qubits must be a tuple of length 3, but is {n_qubits} ({type(n_qubits)})")
        couplings = (n_total_qubits, n_total_qubits)
    elif kind == 'pairwise':
        assert type(n_qubits) == int, f"For kind={kind}, n_qubits must be an integer, but is {n_qubits} ({type(n_qubits)})"
        couplings = (n_qubits, n_qubits)
    elif kind == 'all':
        assert type(n_qubits) == int, f"For kind={kind}, n_qubits must be an integer, but is {n_qubits} ({type(n_qubits)})"
        couplings = (2**n_qubits,)
    else:
        raise ValueError(f"Unknown kind {kind}")

    # if J is not scalar or dict, it must be either the array of the couplings or the limits of the random range
    if not (np.isscalar(J) or isinstance(J, dict)):
        J = np.asarray(J)
        if J.shape == (2,):
            J = np.random.uniform(J[0], J[1], couplings)
        if kind == '1d' and J.shape == (n_qubits, n_qubits):
            # get the offset k=1 diagonal (n_qubits-1 elements)
            idxs = np.where(np.eye(n_qubits, k=1))
            if circular:
                # add the edge element
                idxs = (np.append(idxs[0], 0), np.append(idxs[1], n_qubits-1))
            J = J[idxs]
        if kind == 'pairwise' and len(J.shape) == 1 and len(J) == len(np.triu_indices(n_qubits, k=1)[0]):
            J_mat = np.zeros(couplings)
            J_mat[np.triu_indices(n_qubits, k=1)] = J
            J = J_mat
        assert J.shape == couplings, f"For kind={kind}, J must be a scalar, 2-element vector, or matrix of shape {couplings}, but is {J.shape}"
    elif isinstance(J, dict) and kind != 'all':
        raise ValueError(f"For kind={kind}, J must not be a dict!")

    if h is not None:
        if n_total_qubits != 2 and hasattr(h, '__len__') and len(h) == 2:
            h = np.random.uniform(low=h[0], high=h[1], size=n_total_qubits)
        elif not np.isscalar(h):
            h = np.asarray(h)
        assert np.isscalar(h) or h.shape == (n_total_qubits,), f"h must be a scalar, 2-element vector, or vector of shape {(n_total_qubits,)}, but is {h.shape if not np.isscalar(h) else h}"
    if g is not None:
        if n_total_qubits != 2 and hasattr(g, '__len__') and len(g) == 2:
            g = np.random.uniform(low=g[0], high=g[1], size=n_total_qubits)
        elif not np.isscalar(g):
            g = np.asarray(g)
        assert np.isscalar(g) or g.shape == (n_total_qubits,), f"g must be a scalar, 2-element vector, or vector of shape {(n_total_qubits,)}, but is {g.shape if not np.isscalar(g) else g}"

    # round number to desired precision
    if prec is not None:
        if isinstance(J, dict):
            J = {k: np.round(v, prec) for k, v in J.items()}
        else:
            J = np.round(J, prec)
        h = np.round(h, prec)
        g = np.round(g, prec)

    # generate the Hamiltonian
    H = {}
    # pairwise interactions
    if kind == '1d':
        if np.isscalar(J):
            J = [J]*n_qubits
        for i in range(n_qubits-1):
            if J[i] != 0:
                H['I'*i + 'ZZ' + 'I'*(n_qubits-i-2)] = J[i]
        # last and first qubit
        if circular and n_qubits > 2 and J[-1] != 0:
            H['Z' + 'I'*(n_qubits-2) + 'Z'] = J[-1]
    elif kind == '2d':
        for i in range(n_qubits[0]):
            for j in range(n_qubits[1]):
                # find all 2d neighbors, but avoid double counting
                neighbors = []
                if i > 0:
                    neighbors.append((i-1, j))
                if i < n_qubits[0]-1:
                    neighbors.append((i+1, j))
                if j > 0:
                    neighbors.append((i, j-1))
                if j < n_qubits[1]-1:
                    neighbors.append((i, j+1))
                if circular:
                    if i == n_qubits[0]-1 and n_qubits[0] > 2:
                        neighbors.append((0, j))
                    if j == n_qubits[1]-1 and n_qubits[1] > 2:
                        neighbors.append((i, 0))
                # add interactions
                index_node = i*n_qubits[1] + j
                for neighbor in neighbors:
                    # 1. lower row
                    # 2. same row, but further to the right or row circular (= first column and j is last column)
                    # 3. same column, but column circular (= first row and i is last row)
                    if neighbor[0] > i \
                        or (neighbor[0] == i and (neighbor[1] > j or (j == n_qubits[1]-1 and neighbor[1] == 0 and n_qubits[1] > 2))) \
                        or (neighbor[1] == j and i == n_qubits[0]-1 and neighbor[0] == 0 and n_qubits[0] > 2):
                        index_neighbor = neighbor[0]*n_qubits[1] + neighbor[1]
                        idx1 = min(index_node, index_neighbor)
                        idx2 = max(index_node, index_neighbor)
                        J_val = J[idx1, idx2] if not np.isscalar(J) else J
                        if J_val != 0:
                            H['I'*idx1 + 'Z' + 'I'*(idx2-idx1-1) + 'Z' + 'I'*(n_qubits[0]*n_qubits[1]-idx2-1)] = J_val
    elif kind == '3d':
        for i in range(n_qubits[0]):
            for j in range(n_qubits[1]):
                for k in range(n_qubits[2]):
                    # find all 3d neighbors, but avoid double counting
                    neighbors = []
                    if i > 0:
                        neighbors.append((i-1, j, k))
                    if i < n_qubits[0]-1:
                        neighbors.append((i+1, j, k))
                    if j > 0:
                        neighbors.append((i, j-1, k))
                    if j < n_qubits[1]-1:
                        neighbors.append((i, j+1, k))
                    if k > 0:
                        neighbors.append((i, j, k-1))
                    if k < n_qubits[2]-1:
                        neighbors.append((i, j, k+1))
                    if circular:
                        if i == n_qubits[0]-1 and n_qubits[0] > 2:
                            neighbors.append((0, j, k))
                        if j == n_qubits[1]-1 and n_qubits[1] > 2:
                            neighbors.append((i, 0, k))
                        if k == n_qubits[2]-1 and n_qubits[2] > 2:
                            neighbors.append((i, j, 0))
                    # add interactions
                    index_node = i*n_qubits[1]*n_qubits[2] + j*n_qubits[2] + k
                    for neighbor in neighbors:
                        # 1. lower row
                        # 2. same row, but
                            # a. same layer, but further to the right or row circular (= first column and j is last column)
                            # b. same column, but further behind or layer circular (= first layer and k is last layer)
                        # 3. same column and same layer, but column circular (= first row and i is last row)
                        if neighbor[0] > i \
                            or (neighbor[0] == i and (\
                                (neighbor[2] == k and (neighbor[1] > j or (j == n_qubits[1]-1 and neighbor[1] == 0 and n_qubits[1] > 2))) \
                                or (neighbor[1] == j and (neighbor[2] > k or (k == n_qubits[2]-1 and neighbor[2] == 0 and n_qubits[2] > 2))) \
                            )) \
                            or (neighbor[1] == j and neighbor[2] == k and i == n_qubits[0]-1 and neighbor[0] == 0 and n_qubits[0] > 2):
                            index_neighbor = neighbor[0]*n_qubits[1]*n_qubits[2] + neighbor[1]*n_qubits[2] + neighbor[2]
                            idx1 = min(index_node, index_neighbor)
                            idx2 = max(index_node, index_neighbor)
                            J_val = J[idx1, idx2] if not np.isscalar(J) else J
                            if J_val != 0:
                                H['I'*idx1 + 'Z' + 'I'*(idx2-idx1-1) + 'Z' + 'I'*(n_qubits[0]*n_qubits[1]*n_qubits[2]-idx2-1)] = J_val
    elif kind == 'pairwise':
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                J_val = J[i,j] if not np.isscalar(J) else J
                if J_val != 0:
                    H['I'*i + 'Z' + 'I'*(j-i-1) + 'Z' + 'I'*(n_qubits-j-1)] = J_val
    elif kind == 'all':
        if np.isscalar(J):
            if J == 0:
                raise ValueError("For kind='all', J must not be 0!")
            if n_qubits > 20:
                raise ValueError("Printing out all interactions for n_qubits > 20 is not recommended.")
            for i in range(2, n_qubits+1):
                for membership in itertools.combinations(range(n_qubits), i):
                    H[''.join('Z' if j in membership else 'I' for j in range(n_qubits))] = J
        else: # J is a dict of tuples of qubit indices to interaction strengths
            for membership, strength in J.items():
                if strength != 0:
                    H[''.join('Z' if j in membership else 'I' for j in range(n_qubits))] = strength
    else:
        raise ValueError(f"Unknown kind {kind}")

    # local longitudinal fields
    if np.any(h):
        for i in range(n_total_qubits):
            h_val = h[i] if not np.isscalar(h) else h
            if h_val != 0:
                H['I'*i + 'Z' + 'I'*(n_total_qubits-i-1)] = h_val
    # local transverse fields
    if np.any(g):
        for i in range(n_total_qubits):
            g_val = g[i] if not np.isscalar(g) else g
            if g_val != 0:
                H['I'*i + 'X' + 'I'*(n_total_qubits-i-1)] = g_val

    if as_dict:
        return H

    # convert the Hamiltonian to a string
    # find all unique coefficients (up to precision)
    coeffs_unique = np.unique(list(H.values()))
    # group terms with the same coefficient
    H_rev = {v: [k for k in H if H[k] == v] for v in coeffs_unique}
    # H_groups = [f'{v}*({' + '.join(H_rev[v])})' if len(H_rev[v]) > 1 else f'{v}*{H_rev[v][0]}' for v in reversed(coeffs_unique)]
    H_groups = []
    for v in reversed(coeffs_unique):
        if len(H_rev[v]) > 1:
            H_group = f' + '.join(H_rev[v])
            if v != 1:
                H_group = f'{v}*({H_group})'
        else:
            H_group = H_rev[v][0]
            if v != 1:
                H_group = f'{v}*{H_group}'
        H_groups.append(H_group)
    H_str = ' + '.join(H_groups)

    # Add I*n - I*n if there are no fields
    if H_str == '':
        H_str += 'I'*n_total_qubits + ' - ' + 'I'*n_total_qubits
    # offset
    if np.any(offset):
        H_str += f' + {offset}'
    return H_str

def ising_graph(graph, J=(-1,1), h=(-1,1), g=(-1,1), offset=0):
    """ Takes a graph and generates a Hamiltonian string for it that is compatible with `ph`. """
    import netket as nk # TODO: add networkx, too

    if not isinstance(graph, nk.graph.Graph):
        raise ValueError(f"graph must be a nk.graph.Graph, but is {type(graph)}")

    # get the number of qubits
    n_qubits = graph.n_nodes
    # get the edges
    edges = graph.edges()
    # get the coupling matrix
    J = np.asarray(J)
    if J.shape == ():
        # triangular matrix with all couplings set to J
        J = np.triu(np.ones((n_qubits, n_qubits)), k=1) * J
    elif J.shape == (2,):
        # triangular matrix with all couplings set to a random value in this range
        J = np.triu(np.random.uniform(J[0], J[1], (n_qubits, n_qubits)), k=1)
    elif J.shape == (n_qubits, n_qubits):
        # use the given matrix
        pass
    else:
        raise ValueError(f"J must be a scalar, 2-element vector, or matrix of shape {(n_qubits, n_qubits)}, but is {J.shape}")

    # get the longitudinal fields
    if h is not None:
        h = np.asarray(h)
        if h.shape == ():
            h = np.ones(n_qubits) * h
        elif h.shape == (2,):
            h = np.random.uniform(h[0], h[1], n_qubits)
        elif h.shape == (n_qubits,):
            pass
        else:
            raise ValueError(f"h must be a scalar, 2-element vector, or vector of shape {(n_qubits,)}, but is {h.shape}")

    # get the transverse fields
    if g is not None:
        g = np.asarray(g)
        if g.shape == ():
            g = np.ones(n_qubits) * g
        elif g.shape == (2,):
            g = np.random.uniform(g[0], g[1], n_qubits)
        elif g.shape == (n_qubits,):
            pass
        else:
            raise ValueError(f"g must be a scalar, 2-element vector, or vector of shape {(n_qubits,)}, but is {g.shape}")

    # generate the Hamiltonian
    H_str = ''
    # pairwise interactions
    for i, j in edges:
        if J[i,j] != 0:
            H_str += str(J[i,j]) + '*' + 'I'*i + 'Z' + 'I'*(j-i-1) + 'Z' + 'I'*(n_qubits-j-1) + ' + '
    # local longitudinal fields
    if np.any(h):
        H_str += ' + '.join([str(h[i]) + '*' + 'I'*i + 'Z' + 'I'*(n_qubits-i-1) for i in range(n_qubits) if h[i] != 0]) + ' + '
    # local transverse fields
    if np.any(g):
        H_str += ' + '.join([str(g[i]) + '*' + 'I'*i + 'X' + 'I'*(n_qubits-i-1) for i in range(n_qubits) if g[i] != 0])

    # offset
    if offset != 0:
        H_str += f" + {offset}"

    return H_str

def get_H_energies(H, expi=False, k=None):
    """Returns the energies of the given hamiltonian `H`. For `expi=True` it gives the same result as `get_pe_energies(exp_i(H))` (up to sorting) and for `expi=False` (default) it returns the eigenvalues of `H`."""
    if type(H) == str:
        H = parse_hamiltonian(H, sparse=k is not None)
    if isinstance(H, np.ndarray):
        energies = eigvalsh(H)
        if k is not None:
            energies = energies[...,:k]
    else:
        energies = sp.linalg.eigsh(H, k=k, which='SA', return_eigenvectors=False)[::-1]  # smallest eigenvalues first
    if expi:
        energies = (energies % (2*np.pi))/(2*np.pi)
        energies[energies > 0.5] -= 1
        # energies = np.sort(energies, axis=-1)
    return energies

def print_energies_and_state(H, accuracy=5, r=None, energy_filter=None):
    if isinstance(H, tuple):
        energies, eigvecs = H
    elif is_hermitian(H):
        energies, eigvecs = eigh(H)
    elif is_unitary(H):
        eigvals, eigvecs = np.linalg.eig(H)
        energies = np.angle(eigvals)/(2*np.pi)
    else:
        raise ValueError("`H` must be a Hermitian or unitary matrix or a tuple of energies and eigenvectors!")

    if energy_filter is not None:
        if isinstance(energy_filter, (int, float)):
            mask = np.abs(energies) >= energy_filter
        else:
            assert callable(energy_filter), "energy_filter must be a callable function!"
            mask = energy_filter(energies)
        energies = energies[mask]
        eigvecs = eigvecs[:,mask]

    if r is None:
        print(f"Energy\t\tEigenstate")
        for i, e in enumerate(energies):
            print(f"{e:8.{accuracy}f}\t{unket(eigvecs[:,i])}")
    else:
        print(f"Energy\t\tBinary\t\t\tEigenstate")
        for i, e in enumerate(energies):
            s = binstr_from_float(e, r, complement=True)
            s = " " + s if s[0] != "-" else s
            print(f"{e:8.{accuracy}f}\t{s}\t{unket(eigvecs[:,i])}")
