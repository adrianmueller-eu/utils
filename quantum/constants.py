import warnings
import numpy as np
import itertools
from functools import reduce
from math import sqrt, sin, cos, log2

from ..utils import shape_it
from ..mathlib import matexp, pauli_basis, su, trace_product, generate_recursive
from .state import ket, count_qubits, as_state, is_ket

fs = lambda x: 1/np.sqrt(x)
fs2 = fs(2)

#################
### Unitaries ###
#################

I_ = lambda n: np.eye(2**n)
I, X, Y, Z = su(2, True)
S = np.array([  # matsqrt(Z)
    [1,  0],
    [0, 1j]
])
T_gate = np.array([  # matsqrt(S)
    [1,  0],
    [0,  np.sqrt(1j)]
])
H = H_gate = fs2 * np.array([ # Fourier_matrix(2) = fs2*(X + Z) = 1j*fs2*(Rx(pi) + Rz(pi))
    [1,  1],
    [1, -1]
], dtype=complex)

def R_(gate, theta):
   return matexp(-1j*gate*theta/2)

def Rx(theta):
    ct = cos(theta/2)
    st = -1j*sin(theta/2)
    return np.array([[ct, st], [st, ct]], dtype=complex)
def Ry(theta):  # beam splitter
    ct = cos(theta/2)
    st = sin(theta/2)
    return np.array([[ct, -st], [st, ct]], dtype=complex)
def Rz(theta):  # phase shift
    jt2 = 1j*theta/2
    return np.array([[np.exp(-jt2), 0], [0, np.exp(jt2)]], dtype=complex)

def Rot(phi, theta, lam):
    # return Rz(lam) @ Ry(theta) @ Rz(phi)  # 2x slower
    ct = cos(theta/2)
    st = sin(theta/2)
    ppl = 1j*(phi + lam)/2
    pml = 1j*(phi - lam)/2
    return np.array([
        [np.exp(-ppl)*ct, -np.exp(pml)*st],
        [np.exp(-pml)*st, np.exp(ppl)*ct]
    ])

for i in [2,3]:
    for s, g in zip(itertools.product(['I','X','Y','Z'], repeat=i), itertools.product([I,X,Y,Z], repeat=i)):
        globals()["".join(s)] = reduce(np.kron, g)  # II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
del i, s, g

def C_(A, reverse=False, negative=False):
    if not hasattr(A, 'shape'):
        A = np.asarray(A, dtype=complex)
    n = int(log2(A.shape[0]))
    op0, op1 = [[1,0],[0,0]], [[0,0],[0,1]]
    if negative:
        op0, op1 = op1, op0
    if reverse:
        return np.kron(I_(n), op0) + np.kron(A, op1)
    return np.kron(op0, I_(n)) + np.kron(op1, A)
CNOT = CX = C_(X) # 0.5*(II + ZI - ZX + IX)
XC = C_(X, reverse=True)
CZ = ZC = C_(Z)
CY = C_(Y)
NX = C_(X, negative=True)
XN = C_(X, negative=True, reverse=True)
NZ = C_(Z, negative=True)
ZN = C_(Z, negative=True, reverse=True)
Toffoli = C_(C_(X))
SWAP = np.array([ # 0.5*(XX + YY + ZZ + II), CX @ XC @ CX
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)
iSWAP = np.array([ # 0.5*(1j*(XX + YY) + ZZ + II), R_(XX+YY, -pi/2)
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)
fSWAP = SWAP @ CZ
Fredkin = CSWAP = C_(SWAP)

##############
### States ###
##############

def GHZ_(n):
    a = np.zeros(2**n)
    a[0] = a[-1] = fs2
    return a
Bell = [
    ket('00 + 11'),  # GHZ_(2)
    ket('00 - 11'),
    ket('01 + 10'),
    ket('01 - 10')   # singlet
]
GHZ = GHZ_(3)

Wigner_A = [
    0.5 * (I + X + Y + Z),
    0.5 * (I - X - Y + Z),
    0.5 * (I + X - Y - Z),
    0.5 * (I - X + Y - Z)
]
def get_Wigner_A(i,j,n):
    i_s = [int(x) for x in f'{i:0{n}b}']
    j_s = [int(x) for x in f'{j:0{n}b}']
    return reduce(np.kron, [Wigner_A[2*i_s[k] + j_s[k]] for k in range(n)])

def to_Wigner(state):
    state = np.asarray(state)
    n = count_qubits(state)
    if state.ndim == 1:
        return lambda i,j: (state.conj() @ get_Wigner_A(i,j,n) @ state).real / 2**n
    return lambda i,j: trace_product(state, get_Wigner_A(i,j,n)).real / 2**n

def Wigner_matrix(state, check=2):
    state = as_state(state, check=check)
    n = count_qubits(state)
    if n > 8:
        warnings.warn(f"Generating {2**(2*n)} {2**n}x{2**n} matrices (n = {n}) may take too a long time.", stacklevel=2)
    W = np.zeros((2**n, 2**n))
    As = generate_recursive(Wigner_A, n, Wigner_A, np.kron)
    isket = is_ket(state, print_errors=False)
    for idx, A in enumerate(As):
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

############################
### Non-unitary channels ###
############################

def pauli_channel(p, n=1):
    p = np.asarray(p)
    basis = np.asarray(pauli_basis(n)[1:])
    p0 = p > 1e-12
    p, basis = p[p0], basis[p0]
    return [sqrt(1 - sum(p)) * I_(n), *np.einsum('i,ijk->ijk', np.sqrt(p), basis)]

noise_models = {
    'depolarizing': lambda p: [sqrt(1 - 3*p/4) * I, sqrt(p/4) * X, sqrt(p/4) * Y, sqrt(p/4) * Z],
    'bitflip':      lambda p: [sqrt(1 - p) * I, sqrt(p) * X],
    'phaseflip':    lambda p: [sqrt(1 - p) * I, sqrt(p) * Z],
    'bitphaseflip': lambda p: [sqrt(1 - 2*p/3) * I, np.sqrt(p/3) * X, sqrt(p/3) * Z],
    'pauli':        lambda p, n=1: pauli_channel(p, n),
    'amplitude_damping': lambda p: [np.array([[1, 0], [0, sqrt(1 - p)]]), np.array([[0, sqrt(p)], [0, 0]])],
    'phase_damping':     lambda p: [sqrt(1 - p) * I, np.array([[sqrt(p), 0], [0, 0]]), np.array([[0, 0], [0, sqrt(p)]])],
}
