import numpy as np
from math import sqrt

from .constants import X, Z
from .QuantumComputer import QuantumComputer as QC
from ..mathlib import choice

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