import warnings
import numpy as np
import itertools
from functools import reduce
from math import sqrt, sin, cos, log2

from ..mathlib import matexp, pauli_basis, su
from .state import ket

fs = lambda x: 1/np.sqrt(x)
fs2 = fs(2)

#################
### Unitaries ###
#################

I_ = lambda n, dtype=float: np.eye(2**n, dtype=dtype)
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
    """
    GHZ generalization to n qubits: (|000...0> + |111...1>) / sqrt(2)
    """
    a = np.zeros(2**n)
    a[0] = a[-1] = fs2
    return a
def W_state(n):
    """
    W state of n qubits.
    """
    return sum([ket(2**i, n) for i in range(n)]) / sqrt(n)
Bell = [
    ket('00 + 11'),  # GHZ_(2)
    ket('00 - 11'),
    ket('01 + 10'),
    ket('01 - 10')   # singlet
]
GHZ = GHZ_(3)

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
