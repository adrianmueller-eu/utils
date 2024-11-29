import numpy as np
import itertools
from functools import reduce

from ..mathlib import matexp

fs = lambda x: 1/np.sqrt(x)
f2 = fs(2)
I_ = lambda n: np.eye(2**n)
I = I_(1)
X = np.array([ # 1j*Rx(pi)
    [0, 1],
    [1, 0]
], dtype=complex)
Y = np.array([ # 1j*Ry(pi)
    [0, -1j],
    [1j,  0]
], dtype=complex)
Z = np.array([ # 1j*Rz(pi)
    [1,  0],
    [0, -1]
], dtype=complex)
S = np.array([ # np.sqrt(Z)
    [1,  0],
    [0, 1j]
], dtype=complex)
T_gate = np.array([ # avoid overriding T = True
    [1,  0],
    [0,  np.sqrt(1j)]
], dtype=complex)
H = H_gate = 1/np.sqrt(2) * np.array([ # Fourier_matrix(2) = f2*(X + Z) = 1j*f2*(Rx(pi) + Rz(pi))
    [1,  1],
    [1, -1]
], dtype=complex)

def R_(gate, theta):
   return matexp(-1j*gate*theta/2)

Rx = lambda theta: R_(X, theta)
Ry = lambda theta: R_(Y, theta)
Rz = lambda theta: R_(Z, theta)
for i in [2,3]:
    for s, g in zip(itertools.product(['I','X','Y','Z'], repeat=i), itertools.product([I,X,Y,Z], repeat=i)):
        globals()["".join(s)] = reduce(np.kron, g)  # II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ

def C_(A, reverse=False, negative=False):
    if not hasattr(A, 'shape'):
        A = np.array(A, dtype=complex)
    n = int(np.log2(A.shape[0]))
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