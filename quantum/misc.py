import numpy as np
from math import sqrt
from functools import reduce

from .constants import X, Z
from .QuantumComputer import QuantumComputer as QC
from ..mathlib import choice
from ..quantum import ket, count_qubits, fidelity
from ..plot import imshow_clean

# simulate Bell experiment (CHSH inequality)
def bell_experiment(N=10000):
    Q = Z
    R = X
    S = -(X + Z) / sqrt(2)
    T = -(X - Z) / sqrt(2)
    QS = np.kron(Q, S)
    RS = np.kron(R, S)
    RT = np.kron(R, T)
    QT = np.kron(Q, T)

    qc = QC('01 - 10')
    assert np.isclose(qc.ev(QS) + qc.ev(RS) + qc.ev(RT) - qc.ev(QT), 2*sqrt(2))

    obss = [QS, RS, RT, QT]
    s = {i: 0 for i in range(len(obss))}
    n = {i: 0 for i in range(len(obss))}
    singlet = QC('01 - 10', track_unitary=False, check=0)
    for _ in range(N):
        qc = singlet.copy()
        i = choice(4)
        s[i] += qc.measure(obs=obss[i], return_as='energy')
        n[i] += 1
    return s[0]/n[0] + s[1]/n[1] + s[2]/n[2] - s[3]/n[3]

# estimate the fidelity between two states using the swap test
def swap_test(psi='0', phi='0 + 1', N=1000):
    psi = ket(psi)
    phi = ket(phi)
    assert psi.shape == phi.shape, f"{psi.shape} != {phi.shape}"
    print("Actual fidelity:", fidelity(psi, phi))
    n = count_qubits(psi)
    state = reduce(np.kron, [ket(0), psi, phi])
    qc = QC(state)
    print(qc)
    def perform_swap_test(qc):
        qc.h(0)
        for j in range(n):
            qc.cswap(0, j+1, j+n+1)
        qc.h(0)
        return qc.measure(0, return_as=int)

    # perform N swap tests
    outcomes = [perform_swap_test(qc.copy()) for _ in range(N)]
    return 1 - 2*np.mean(outcomes)

def the_arrow(n=1, figsize=None):
    arrow = np.array([
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1]], dtype='i1')*2-1
    m = reduce(np.kron, [arrow]*n)
    if figsize is None:
        fs = max(1, 2**(2*n-4.68188))
        figsize = (fs, fs)
    imshow_clean(~m, figsize, cmap='hot')

# TODO: Grover's algorithm
# TODO: number factorization (Shor's algorithm)
# TODO: HHL algorithm
