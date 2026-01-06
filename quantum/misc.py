import numpy as np
from math import sqrt, ceil, log2
from functools import reduce

from .constants import X, Z
from .QuantumComputer import QuantumComputer as QC
from ..mathlib import choice
from ..quantum import ket, count_qubits, fidelity
from ..plot import imshow_clean
from ..utils import is_int

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
    singlet = QC('01 - 10', track_operators=False, check=0)
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

def grover(key, n=None):
    """
    Grover's algorithm for searching an unsorted database.

    Parameters
    ----------
    key : int
        The key to search for.
    n : int
        The number of qubits in the database.

    Returns
    -------
    QC
        The quantum circuit for Grover's algorithm.
    result: str
        The result of the measurement.
    """
    assert is_int(key) and key > 0, f"Invalid key: {key}"
    if n is None:
        n = ceil(log2(key))
    assert is_int(key) and is_int(n) and n > 0, f"Invaoid input: key={key}, n={n}"
    assert 0 <= key < 2**n, f"Too few qubits: {key} >= {2**n}"
    qc = QC(n)
    reps = ceil(np.pi/4*1/np.arcsin(1/sqrt(2**n)))  # where 1/np.arcsin(1/sqrt(2**n)) ≈ sqrt(2**n)
    qc.h(range(n))
    for i in range(reps):
        raise NotImplementedError("Grover's algorithm is not implemented yet.")
    return qc.measure()

# TODO: number factorization (Shor's algorithm)
# TODO: HHL algorithm
