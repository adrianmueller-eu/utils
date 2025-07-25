from .state import *
from .constants import *
from .hamiltonian import *
from .unitary import *
from .info import *
from .QuantumComputer import *
from .misc import *

###############
### Aliases ###
###############

ph = parse_hamiltonian
pu = parse_unitary
QC = QuantumComputer

qc = QuantumComputer()

#############
### Tests ###
#############

from numpy.random import randint
from ..mathlib import is_involutory, anticommute, float_from_binstr, random_hermitian, random_unitary, trace_product

def test_quantum_all():
    tests = [
        _test_constants,
        _test_fourier_matrix,
        _test_parse_unitary,
        _test_subunitary,
        _test_parse_hamiltonian,  # required by _test_random_ham
        _test_random_ham,  # required by _test_exp_i
        _test_exp_i,
        _test_get_H_energies_eq_get_pe_energies,
        _test_channels,
        _test_random_ket,  # required _test_reverse_qubit_order
        _test_random_dm,   # required by _test_partial_trace
        _test_reverse_qubit_order,
        _test_partial_trace,
        _test_QuantumComputer,
        _test_ket_unket,
        _test_op_dm,
        _test_is_ket,
        _test_is_dm,
        _test_ensemble,
        _test_is_eigenstate,
        _test_count_qubits,
        _test_Wigner,
        _test_von_neumann_entropy,
        _test_entanglement_entropy,
        _test_fidelity,
        _test_trace_distance,
        _test_schmidt_decomposition,
        _test_purify,
        _test_correlation_quantum,
        _test_ground_state,
        _test_ising,
        _test_pauli_basis,
        _test_pauli_decompose
    ]

    for test in tests:
        print("Running", test.__name__, "... ", end="", flush=True)
        test()
        print("Test succeeded!", flush=True)

def _test_constants():
    global I, X, Y, Z, H_gate, S, T_gate, CNOT, SWAP
    H = H_gate

    assert is_involutory(X)
    assert is_involutory(Y)
    assert is_involutory(Z)
    assert is_involutory(H)
    assert anticommute(X,Y)
    assert anticommute(Y,Z)
    assert anticommute(X+Y-Z,H)
    assert np.allclose(S @ S, Z)
    assert np.allclose(T_gate @ T_gate, S)
    assert is_involutory(CNOT)
    assert is_involutory(SWAP)
    assert np.allclose(Rx(2*np.pi), -I)
    assert np.allclose(Ry(2*np.pi), -I)
    assert np.allclose(Rz(2*np.pi), -I)
    angle = np.random.rand()*4*np.pi - 2*np.pi
    assert np.allclose(Rx(angle), R_(X, angle))
    assert np.allclose(Ry(angle), R_(Y, angle))
    assert np.allclose(Rz(angle), R_(Z, angle))
    l,t,p = np.random.rand(3)*4*np.pi - 2*np.pi
    assert np.allclose(Rot(p,t,l), Rz(l) @ Ry(t) @ Rz(p))

def _test_fourier_matrix():
    assert np.allclose(Fourier_matrix(2), H)
    n = randint(2,8)
    F = Fourier_matrix(2**n)
    assert is_unitary(F)
    assert np.allclose(F @ ket('0'*n), normalize(np.ones(2**n)))  # Fourier creates a full superposition or ...
    assert np.allclose(F[:,0], parse_unitary('H'*n)[:,0])  # ... in other words

def _test_parse_unitary():
    assert np.allclose(parse_unitary('I'), I)
    assert np.allclose(parse_unitary('X'), X)
    assert np.allclose(parse_unitary('Y'), Y)
    assert np.allclose(parse_unitary('Z'), Z)
    assert np.allclose(parse_unitary('T'), T_gate)
    assert np.allclose(parse_unitary('t'), T_gate.T.conj())
    assert np.allclose(parse_unitary('S'), S)
    assert np.allclose(parse_unitary('s'), S.T.conj())

    assert np.allclose(parse_unitary('CX'), CX)
    assert np.allclose(parse_unitary('XC'), reverse_qubit_order(CX))
    assert np.allclose(parse_unitary('CCX'), Toffoli)
    assert np.allclose(parse_unitary('XCC'), reverse_qubit_order(Toffoli))
    assert np.allclose(parse_unitary('CX @ XC @ CX'), SWAP)
    assert np.allclose(parse_unitary('SS @ HI @ CX @ XC @ IH'), iSWAP)
    assert np.allclose(parse_unitary('IIII @ IIII'), I_(4))
    U_actual = reorder_qubits(parse_unitary('IXZ'), [0,2,1])
    assert np.allclose(U_actual, parse_unitary('IZX'))

    assert is_unitary(parse_unitary('XCX'))
    assert is_unitary(parse_unitary('CXC'))
    assert is_unitary(parse_unitary('XCXC'))
    assert is_unitary(parse_unitary('XCXCX @ CZCZC'))

    assert np.sum(np.where((parse_unitary('XXX') - I_(3)) != 0)) == 112
    assert np.sum(np.where((parse_unitary('CXX') - I_(3)) != 0)) == 88
    assert np.sum(np.where((parse_unitary('XCX') - I_(3)) != 0)) == 72
    assert np.sum(np.where((parse_unitary('XXC') - I_(3)) != 0)) == 64
    assert np.sum(np.where((parse_unitary('CCX') - I_(3)) != 0)) == 52
    assert np.sum(np.where((parse_unitary('CXC') - I_(3)) != 0)) == 48
    assert np.sum(np.where((parse_unitary('XCC') - I_(3)) != 0)) == 40
    assert np.sum(np.where((parse_unitary('CCC') - I_(3)) != 0)) == 0
    pass

def _test_subunitary():
    U1 = random_unitary(2**2)
    U2 = random_unitary(2**3)
    U = np.kron(U1, U2)
    assert is_separable_unitary(U, [0,1])
    assert not is_separable_unitary(U, [1])
    assert not is_separable_unitary(U, [2,1])
    assert not is_separable_unitary(U, [0,1,2])
    U1_ = get_subunitary(U, [0,1])
    phases = U1/U1_
    assert np.allclose(phases, phases[0,0]), phases

def _test_parse_hamiltonian():
    H = parse_hamiltonian('0.5*(II + ZI - ZX + IX)')
    assert np.allclose(H, CNOT)

    H = parse_hamiltonian('0.5*(XX + YY + ZZ + II)')
    assert np.allclose(H, SWAP)

    H = parse_hamiltonian('-(XX + YY + .5*ZZ) + 1.5')
    assert np.allclose(np.sum(H), 2)

    H = parse_hamiltonian('0.2*(-0.5*(3*XX + 4*YY) + 1*II)')
    assert np.allclose(np.sum(H), -.4)

    H = parse_hamiltonian('X + 2*(I+Z)')
    assert np.allclose(H, X + 2*(I+Z))

    H = parse_hamiltonian('1*(ZZI + IZZ) + 1*(ZII + IZI + IIZ)')
    assert np.allclose(np.sum(np.abs(H)), 14)

    H = parse_hamiltonian('-.25*(ZZI + IZZ) + 1.5')
    assert np.allclose(np.sum(np.abs(H)), 12)

    H = parse_hamiltonian('1.2*IZZI')
    IZZI = np.kron(np.kron(I, Z), np.kron(Z, I))
    assert np.allclose(H, 1.2*IZZI)

def _test_random_ham():
    n_qubits = np.random.randint(1, 5)
    n_terms = np.random.randint(1, 100)
    n_terms = min(n_terms, 2**(2*n_qubits)-1)
    H = random_ham(n_qubits, n_terms)
    H = parse_hamiltonian(H)
    assert H.shape == (2**n_qubits, 2**n_qubits)
    assert np.allclose(np.trace(H), 0)
    assert is_hermitian(H)

def _test_exp_i():
    n = randint(1,6)
    n_terms = randint(1, 2**(n+1))
    H_str = random_ham(n, n_terms)
    H = ph(H_str)
    U_expect = matexp(1j*H)
    U_actual = get_unitary(exp_i(H))
    assert np.allclose(U_actual, U_expect), f"H_str: {H_str}"
    U_actual = exp_i(H).get_unitary()
    assert np.allclose(U_actual, U_expect), f"H_str: {H_str}"

    U_expect = matexp(1j*3*H)
    U_actual = get_unitary(exp_i(H)**3)
    assert np.allclose(U_actual, U_expect), f"H_str: {H_str}"
    U_actual = (exp_i(H)**3).get_unitary()
    assert np.allclose(U_actual, U_expect), f"H_str: {H_str}"
    U_actual = exp_i(H).get_unitary(3)
    assert np.allclose(U_actual, U_expect), f"H_str: {H_str}"
    U_actual = exp_i(H, k=3).get_unitary()
    assert np.allclose(U_actual, U_expect), f"H_str: {H_str}"

def _test_get_H_energies_eq_get_pe_energies():
    n_qubits = np.random.randint(1, 5)
    n_terms = np.random.randint(1, 100)
    n_terms = min(n_terms, 2**(2*n_qubits)-1)
    H = random_ham(n_qubits, n_terms, scaling=False)
    H = parse_hamiltonian(H)

    A = np.sort(get_pe_energies(exp_i(H)))
    B = np.sort(get_H_energies(H, expi=True))
    assert np.allclose(A, B), f"{A} ≠ {B}"

def _test_ket_unket():
    assert np.allclose(ket('0'), [1,0])
    assert np.allclose(ket('1'), [0,1])
    assert np.allclose(ket(0), [1,0])
    assert np.allclose(ket(2), [0,0,1,0])
    assert np.allclose(ket('00 + 11 + 01 + 10'), [.5,.5,.5,.5])
    assert np.allclose(ket('00+ 11- 01 -10'), [.5,-.5,-.5,.5])
    assert np.allclose(ket('00 + 11'), ket('2*00 + 2*11'))
    # assert np.allclose(ket('00 - 11'), ket('2*(00 - 11)'))  # parentheses NYI
    assert np.allclose(ket('3*00 + 4*01 + 5*11'), [np.sqrt(9/50), np.sqrt(16/50), 0, np.sqrt(25/50)])
    assert unket(ket('1010')) == '1010'
    assert unket(ket(10)) == '1010'
    assert unket(ket(10, 5)) == '01010'
    # ket should be fast enough to return already-kets 1000 times in negligible time
    import time
    psi = random_ket(2)
    max_time = 0.01*2  # remove *2 when setting expected iterations
    start = time.time()

    for _ in range(70):
        assert time.time() - start < max_time, f"ket is too slow (iteration {_}/70)"
        for _ in range(100):
            ket(psi, check=0, renormalize=False)

    start = time.time()
    for _ in range(25):
        assert time.time() - start < max_time, f"ket is too slow (iteration {_}/25)"
        for _ in range(100):
            ket(psi, check=1, renormalize=False)

def _test_random_ket():
    n_qubits = np.random.randint(1, 10)
    psi = random_ket(n_qubits)
    assert psi.shape == (2**n_qubits,)
    assert np.isclose(np.linalg.norm(psi), 1)
    kets = random_ket(5, 1)
    assert kets.shape == (1, 2**5)
    assert np.isclose(np.linalg.norm(kets[0]), 1)
    kets = random_ket(2, 1000)
    assert kets.shape == (1000, 2**2)
    assert np.allclose(np.linalg.norm(kets, axis=1), 1)
    # check the kets are haar distributed
    assert allclose0(np.mean(kets, axis=0), tol=0.05)

def _test_op_dm():
    assert np.allclose(op(0),   [[1,0], [0,0]])
    assert np.allclose(dm(0),   [[1,0], [0,0]])
    assert np.allclose(op(0,1), [[0,1], [0,0]])
    assert np.allclose(op(1,0), [[0,0], [1,0]])
    assert np.allclose(op(1),   [[0,0], [0,1]])
    assert np.allclose(dm(1),   [[0,0], [0,1]])
    O = op(0,3,n=2)
    assert O.shape[0] == O.shape[1]
    # dm should be fast enough to return already-density-matrices 1000 times in negligible time
    import time
    rho = random_dm(2)
    max_time = 0.01*2  # remove *2 when setting expected iterations

    for i, n, level in zip(range(4), [5,10,30,38], [3,2,1,0]):
        start = time.time()
        for _ in range(n):
            assert time.time() - start < max_time, f"dm is too slow ({i}, iteration {_}/{n})"
            for _ in range(100):
                dm(rho, check=level, renormalize=False)

def _test_is_ket():
    assert is_ket([1,0,0,0])
    assert not is_ket([1,0,0,1], print_errors=False)
    assert not is_ket([1,0,0,0,0], print_errors=False)
    warnings.filterwarnings("ignore")
    assert is_ket([1])  # 0-qubit state
    assert not is_ket([[1]], print_errors=False)
    warnings.filterwarnings("default")
    assert is_ket('0.5*00 + 11')
    assert not is_ket('abc', print_errors=False)
    assert not is_ket(np.eye(2)/2, print_errors=False)
    assert is_ket(random_ket(2))

def _test_is_dm():
    assert is_dm(I_(2)/2**2)
    assert not is_dm(I_(2), print_errors=False)
    assert not is_dm(np.eye(3), print_errors=False)
    warnings.filterwarnings("ignore")
    assert is_dm([[1]])  # 0-qubit state
    warnings.filterwarnings("default")
    assert is_dm('0.5*00 + 11')
    assert not is_dm('abc', print_errors=False)
    assert not is_dm([1], print_errors=False)
    assert not is_dm([1,0,0,0], print_errors=False)
    assert is_dm(random_dm(2))

    # random Bloch vector
    v = np.random.uniform(-1, 1, 3)
    if np.linalg.norm(v) > 1:  # also test mixed states
        v = normalize(v)
    rho = (I + v[0]*X + v[1]*Y + v[2]*Z)/2
    assert is_dm(rho)

def _test_ensemble():
    n = np.random.randint(1, 5)
    rho = random_dm(n)
    p, kets = ensemble_from_state(rho)
    assert is_dm(dm_from_ensemble(p, kets))
    ket = random_ket(n)
    p, kets = ensemble_from_state(ket)
    assert np.isclose(p[0], 1)
    assert len(kets) == 1
    assert len(kets[0]) == 2**n

def _test_random_dm():
    n_qubits = np.random.randint(1, 5)
    rho = random_dm(n_qubits)
    assert is_dm(rho)
    rho = random_dm(6, 42)
    D = np.linalg.eigvalsh(rho)
    assert is_psd(rho, D)
    assert np.sum(D > 1e-12) == 42
    rhos = random_dm(2, 3, size=(3,2))
    assert rhos.shape == (3,2,4,4)

def _test_Wigner():
    W = Wigner_from_state(ket('00'))
    assert np.allclose(W, [[0.25]*4, [0]*4, [0]*4, [0]*4])
    rho = random_dm(4)
    W = Wigner_from_state(rho)
    assert np.isclose(W.sum(), 1), f"Wigner matrix is not normalized: {W.sum()}"
    rho_reconstructed = dm_from_Wigner(W)
    assert np.allclose(rho, rho_reconstructed), f"rho = {rho}\nrho_reconstructed = {rho_reconstructed}"

def _test_QuantumComputer():
    # test basic functionality
    qc = QuantumComputer(track_operators=False)  # remove() would turn it into density matrix mode, but we want to keep state vector here
    qc.x(0)
    assert unket(qc.get_state()) == '1', f"{unket(qc.get_state())} ≠ '1'"
    qc.cx(0, 1)
    assert unket(qc.get_state()) == '11', f"{unket(qc.get_state())} ≠ '11'"
    qc.reset()
    assert unket(qc.get_state()) == '00', f"{unket(qc.get_state())} ≠ '00'"
    qc.init('00 + 11')
    outcome = qc.resetv([1])
    assert outcome in ['0', '1']
    qc.resetv(0)
    assert unket(qc.get_state()) == '00', f"{unket(qc.get_state())} ≠ '00'"
    qc.h()
    qc.x(2)
    qc.init(2, [0,1])
    assert unket(qc.get_state()) == '101', f"{unket(qc.get_state())} ≠ '101'"
    qc.z([0,2])  # noop
    qc.swap(1,2)
    assert unket(qc.get_state()) == '110', f"{unket(qc.get_state())} ≠ '110'"
    qc.reset(1)
    assert unket(qc.get_state()) == '100', f"{unket(qc.get_state())} ≠ '100'"
    qc.remove([0,2])
    assert unket(qc.get_state()) == '0', f"{unket(qc.get_state())} ≠ '0'"
    qc.reset(1)
    qc.x(2)
    result = qc.measure('all')
    assert result == '01'
    qc = QuantumComputer(2)
    qc.reset([1,2])
    assert qc.n == 3, f"qc.n = {qc.n} ≠ 3"
    qc.copy()
    qc1 = QC(state=1)
    assert qc1.qubits == (0,), f"{qc1.qubits} ≠ (0,)"

    k1 = random_ket(2)
    k2 = random_ket(3)
    qc = QC(np.kron(k1, k2))
    qc.remove([0,1])
    assert np.allclose(abs(qc.get_state().conj() @ k2), 1), f"{qc.get_state()} ≠ {k2}"

    qc = QC(2)
    qc[:2] = random_dm(2)
    assert qc.n == 2, qc._qubits
    assert qc.get_state().shape == (4,4), qc._state.shape
    qc.remove(1)
    qc[-1] = '0'
    assert qc.is_pure()
    assert qc._operators[0].shape == (2,4), qc._operators

    # Heisenberg uncertainty principle
    qc = QuantumComputer(1, 'random')
    assert qc.std(X) * qc.std(Z) >= abs(qc.ev(1j*(X@Z - Z@X)))/2

    # test functions
    global XX, YY
    qc = QuantumComputer(2)
    qc.h(0)
    qc.cx(0, 1)
    assert np.allclose(qc.get_state(), normalize([1,0,0,1]))  # Bell state |00> + |11>
    qc(I, 0)  # let it reshape to [2,2]
    assert np.allclose(qc[0], I/2)
    assert np.allclose(qc[1], I/2)
    assert np.isclose(qc.purity(0), 0.5)
    p = qc.probs(obs=XX)  # Bell basis is the basis of XX / YY observable -> deterministic outcome
    assert np.isclose(entropy(p), 0), f"p = {p}"
    with qc.observable(YY):  # test context manager
        p = qc.probs()
        assert np.isclose(entropy(p), 0), f"p = {p}"
    U_expected = parse_unitary('CX @ HI')  # check generated unitary
    assert np.allclose(qc.get_unitary(), U_expected), f"Incorrect unitary:\n{qc.get_unitary()}\n ≠\n{U_expected}"

    assert np.isclose(qc.std(X, 0), 1)
    assert np.isclose(qc.std(Y, 0), 1)
    assert np.isclose(qc.std(Z, 0), 1)
    assert np.isclose(qc.ev(X, 0), 0)
    assert np.isclose(qc.ev(Y, 0), 0)
    assert np.isclose(qc.ev(Z, 0), 0)

    assert np.isclose(qc.entanglement_entropy(1), 1)
    assert np.isclose(qc.correlation(0, 1, Z, Z), 1)
    S = qc.schmidt_decomposition(0, coeffs_only=True)
    assert np.allclose(S, [fs2, fs2])
    assert np.isclose(qc.mutual_information(0, 1), 2)

    # test density matrix
    qc.decohere(0)
    assert np.isclose(qc.mutual_information(0, 1), 1)
    assert qc.is_matrix_mode()
    assert np.allclose(qc.get_state(), (dm('00') + dm('11'))/2)
    qc.purify()
    assert np.allclose(qc.get_state(), ket('000 + 111'))  # purification only needs one ancilla qubit in this case
    assert not qc.is_matrix_mode()
    assert qc[:].shape == (8,8)
    assert not qc.is_unitary()
    assert qc.is_isometric()
    qc.to_dm()
    rdm0 = qc.get_state([0], collapse=True)
    assert rdm0.shape == (2,2), f"{rdm0.shape} ≠ (2,2)"
    qc.remove([0])
    qc.remove(1)
    QC(random_dm(3, rank=3)).purify()

    qc = QuantumComputer('00 + 01')
    assert np.allclose(qc.schmidt_coefficients([1]), [1])
    rdm0 = qc.to_dm().get_state(0, collapse=True)
    assert_dm(rdm0, n=1)
    qc = QC('00 + 01 + 11')
    assert np.isclose(trace_product(qc[0], qc[0]), 7/9)
    assert np.isclose(trace_product(qc[1], qc[1]), 7/9)
    qc = QC(3, 'random')
    assert not qc.is_matrix_mode()
    assert np.isclose(np.sum(qc.schmidt_coefficients([0])**2), 1)

    assert qc.is_unitary()
    U = random_unitary(2**2)
    qc(U, [0,2])
    assert qc.is_unitary([0,1,2])
    assert not qc.is_unitary([0,1])
    assert qc.is_unitary([0,2])
    # E = random_channel(1)
    # qc(E, 1)
    # assert not qc.is_unitary(1)  # not implemented yet
    # assert qc.is_unitary([0,2])

    # more complex test
    qc = QuantumComputer(15)
    U = parse_unitary('XYZCZYX')
    qc(U, choice(qc.qubits, 7, False))
    qc.remove([5,7,6,10,2])
    U = random_unitary(2**5)
    qc(U, choice(qc.qubits, 5, False))
    assert qc.n == 10

    # test phase estimation
    H = ph(f'{1/8}*(IZ + ZI + II)')
    U = exp_i(2*np.pi*H)
    assert np.isclose(np.trace(H @ dm('00')), float_from_binstr('.011'))  # 00 is eigenstate with energy 0.375 = '011'
    state_qubits = ['s0', 's1']
    qc = QuantumComputer(state_qubits)
    E_qubits = ['e0', 'e1', 'e2']
    qc.pe(U, state_qubits, E_qubits)
    res = qc.measure(E_qubits)
    assert res == '011', f"measurement result was {res} ≠ '011'"

    qc.remove('s0')
    assert np.allclose(qc.get_state(), ket('0011'))
    qc.remove('all')
    qc.add(0)

    # test schmidt decomposition
    qc = QuantumComputer(5, 'random')
    bip = choice([i for i,o in bipartitions(range(5))])
    S, U, V = qc.schmidt_decomposition(bip)
    # check RDM for subsystem A
    rho_expect = qc[bip]
    rho_actual = np.sum([l_i**2 * np.outer(A_i, A_i.conj()) for l_i, A_i in zip(S, U)], axis=0)
    assert np.allclose(rho_actual, rho_expect), f"rho_expect - rho_actual = {rho_expect - rho_actual}"
    # check RDM for subsystem B
    rho_expect = qc[[i for i in qc.qubits if i not in bip]]
    rho_actual = np.sum([l_i**2 * np.outer(B_i, B_i.conj()) for l_i, B_i in zip(S, V)], axis=0)
    assert np.allclose(rho_actual, rho_expect), f"rho_expect - rho_actual = {rho_expect - rho_actual}"

    # test density matrix
    qc = QuantumComputer('0100 + 1010')
    assert np.isclose(qc.entanglement_entropy(3), 0)
    qc.remove(3)
    assert not qc.is_matrix_mode()
    qc.remove(1)
    assert qc.is_matrix_mode()
    qc = QuantumComputer('0100 + 1010')
    qc.KEEP_VECTOR = False
    qc.remove(3)
    assert qc.is_matrix_mode()
    assert np.isclose(qc.entanglement_entropy(1), 1)
    qc.remove(1)
    qc.x(2)
    assert np.allclose(qc.get_state(), (dm('01') + dm('10'))/2)
    qc.to_ket(kind='sample')

    # test kraus operators
    qc = QuantumComputer(2)
    qc.h(0)
    qc.noise('depolarizing', 0)
    ops = qc.get_operators()
    assert len(ops) == 4
    assert_kraus(ops, n=(2,2))
    assert np.allclose(QC(2)(ops)[:], qc[:])

    qc.cx(0,1)
    qc.measure(0, collapse=False)
    ops = qc.get_operators()
    assert len(ops) == 8, len(ops)
    assert np.allclose(qc[:], QC(2)(ops)[:])
    qc.reset(0, collapse=False)
    ops = qc.get_operators()
    assert np.allclose(qc[:], QC(2)(ops)[:])

    # test choi matrix / superoperator
    qc.compress_operators()
    assert len(qc.get_operators()) == 4

    qc = QC(1, as_superoperator=True)
    assert qc.is_matrix_mode()
    choi_actual = qc.choi_matrix()
    if hasattr(choi_actual, 'toarray'):
        choi_actual = choi_actual.toarray()
    choi_expected = np.array([[1, 0, 0, 1],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [1, 0, 0, 1]])
    assert np.allclose(choi_actual, choi_expected), f"choi_actual = {choi_actual}\nchoi_expected = {choi_expected}"
    qc(X)
    choi_actual = qc.choi_matrix()
    if hasattr(choi_actual, 'toarray'):
        choi_actual = choi_actual.toarray()
    choi_expected = np.array([[0, 0, 0, 0],
                              [0, 1, 1, 0],
                              [0, 1, 1, 0],
                              [0, 0, 0, 0]])
    assert np.allclose(choi_actual, choi_expected), f"choi_actual = {choi_actual}\nchoi_expected = {choi_expected}"
    ops = noise_models['depolarizing'](0.1)
    qc(ops)
    qc.remove(0)
    qc.add(0)
    qc = QC(2, as_superoperator=True)
    U = random_unitary(2**2)
    qc(U)
    C1 = qc.choi_matrix()
    qc = QC(2, as_superoperator=False)
    qc(U)
    C2 = qc.choi_matrix()
    assert np.allclose(C1, C2), f"C1 = {C1}\nC2 = {C2}"
    qc(X, 1)  # partial application

    # test n_out > n_in
    qc = QC(1)
    qc.init(0, [1], collapse=False)
    qc(random_unitary(2**2))
    qc.measure([1], collapse=False)
    qc.compress_operators()

    if "qiskit" in sys.modules:
        warnings.filterwarnings("ignore")  # ignore deprecation warnings
        from qiskit.circuit.library import phase_estimation, RYGate, QFTGate
        U = QuantumCircuit(2)  # RYGate(np.pi/12) fails for pi/(2**(n-2)) for n >= 3
        U.rx(0.1, 0)
        n = randint(1,8)
        s_reg, e_reg = [0,1], list(range(2, n+2))
        U_PE1 = get_unitary(phase_estimation(n, U))
        U_PE1 = reorder_qubits(U_PE1, (e_reg + s_reg)[::-1])  # qiskit outputs qubits backwards
        qc2 = QC(2+n, track_operators=True).pe(get_unitary(U), s_reg[::-1], e_reg)
        U_PE2 = qc2.get_unitary()
        assert np.allclose(U_PE1, U_PE2)

        U_QFT1 = get_unitary(QFTGate(n))
        U_QFT2 = get_unitary(QC(n, track_operators=True).qft(range(n)))
        assert np.allclose(U_QFT1, U_QFT2)

        def check_apply_qiskit_circuit(qc, qubits='all'):
            U1 = reverse_qubit_order(get_unitary(qc))

            for elementary in [True, False]:
                qc2 = QC()
                qc2.apply_qiskit_circuit(qc, qubits, include_phases=True, elementary_only=elementary)
                U2 = get_unitary(qc2)
                assert np.allclose(U1, U2), np.linalg.norm(U1 - U2)

            qc2 = QC()
            qc2.apply_qiskit_circuit(qc, include_phases=False, elementary_only=True)
            U3 = get_unitary(qc2)
            qc2 = QC()
            qc2.apply_qiskit_circuit(qc, include_phases=False, elementary_only=False)
            U4 = get_unitary(qc2)
            assert np.allclose(U3, U4), np.linalg.norm(U3 - U4)

        qc1 = QuantumCircuit(2)
        qc1.u(0.1, 0.2, 0.3, 0)
        qc1.h(0)
        qc1.cx(0,1)
        check_apply_qiskit_circuit(qc1)

        qc1 = QuantumCircuit(2, 4)
        qc1.x(0)  # state '10'
        qc1.measure([0,1], [0,1])
        qc1.x([0,1])  # state '01'
        qc1.measure([0,1], [2,3])
        qc2 = QC()
        outcome = qc2.apply_qiskit_circuit(qc1)
        assert outcome == '1001', f"Outcome was: {outcome}"

        from qiskit.circuit.library import QFT
        check_apply_qiskit_circuit(QFT(4, do_swaps=False))
        check_apply_qiskit_circuit(QFT(4, do_swaps=True))
        warnings.filterwarnings("default")


def _test_channels():
    # POVM
    assert np.allclose(measurement_operator(0, 2), op(0, n=2))
    assert np.allclose(measurement_operator(3, 3, [2,1], as_matrix=False), [0,0,0,1]*2)
    assert np.allclose(sum(POVM(2)), np.eye(2**2))
    povm = POVM(2, [0], as_matrix=True)
    assert len(povm) == 2, f"len(povm) = {len(povm)} ≠ 2"
    assert np.allclose(povm[0], op('00') + op('01')), f"povm[0] = {povm[0]} ≠ op('00') + op('01')"
    assert np.allclose(povm[1], op('10') + op('11')), f"povm[1] = {povm[1]} ≠ op('10') + op('11')"
    assert_kraus(povm, n=(2,2))

    # combine channels
    K1 = [pu('HI')]
    K2 = [pu('CX')]
    K3 = combine_channels(K2, K1)
    assert len(K3) == 1
    assert np.allclose(K3, pu('CX @ HI'))

    # removal / extension
    def _test(st, n):
        K_r = removal_channel(n)
        assert_kraus(K_r, n=(0,n))
        K_e = extension_channel(st, n)
        assert_kraus(K_e, n=(n,0))
        K_c = combine_channels(K_r, K_e)
        assert_kraus(K_c, n=(0,0))
        assert K_c[0].shape == (1,1)
        K_c1 = combine_channels(K_e, K_r)
        K_c2 = reset_channel(st, n)
        assert_kraus(K_c2, n=(n,n))
        assert np.allclose(K_c1, K_c2)
        return K_c2

    n = 3
    _test(0, n)
    _test(random_ket(n), n)
    _test(dm(0, n=n), n)
    _test(random_dm(n), n)

    # random_channel
    n = randint(0, 5)
    E = random_channel(n)
    assert_kraus(E)
    assert len(E) == 2**n, f"{len(E)} ≠ {2**n}"
    n_in = randint(0, 5)
    E = random_channel(n,n_in)
    assert_kraus(E)
    assert len(E) == 2**n_in, f"{len(E)} ≠ {2**n_in}"

    # partial operation
    U = Fourier_matrix(2**5)
    subsystem_state = random_ket(2)
    env_state = random_ket(3)
    Ks = partial_operation(U, [0,1], env_state)
    assert_kraus(Ks, n=(2,2))
    st_expect = QC(np.kron(subsystem_state, env_state))(U)[0,1]  # partial_trace(U @ np.kron(subsystem_state, env_state), [0,1])
    st_actual = QC(subsystem_state)(Ks)[0,1]  # sum([K @ dm(subsystem_state) @ K.conj().T for K in Ks])
    assert np.allclose(st_actual, st_expect), f"Partial operation failed: {st_actual} != {st_expect}"

def _test_reverse_qubit_order():
    # known 3-qubit matrix
    psi = np.kron(np.kron([1,1], [0,1]), [1,-1])
    psi_rev1 = np.kron(np.kron([1,-1], [0,1]), [1,1])
    psi_rev2 = reverse_qubit_order(psi)
    assert np.allclose(psi_rev1, psi_rev2)

    # same as above, but with n random qubits
    n = 10
    psis = [random_ket(1) for _ in range(n)]
    psi = psis[0]
    for i in range(1,n):
        psi = np.kron(psi, psis[i])
    psi_rev1 = psis[-1]
    for i in range(1,n):
        psi_rev1 = np.kron(psi_rev1, psis[-i-1])

    psi_rev2 = reverse_qubit_order(psi)
    assert np.allclose(psi_rev1, psi_rev2)

    # general hamiltonian
    H = parse_hamiltonian('IIIXX')
    H_rev1 = parse_hamiltonian('XXIII')
    H_rev2 = reverse_qubit_order(H)
    assert np.allclose(H_rev1, H_rev2)

    H = parse_hamiltonian('XI + YI')
    H_rev1 = parse_hamiltonian('IX + IY')
    H_rev2 = reverse_qubit_order(H)
    assert np.allclose(H_rev1, H_rev2), f"{H_rev1} \n≠\n {H_rev2}"

    # pure density matrix
    psi = np.kron(np.kron([1,1], [0,1]), [1,-1])
    rho = np.outer(psi, psi)
    psi_rev1 = np.kron(np.kron([1,-1], [0,1]), [1,1])
    rho_rev1 = np.outer(psi_rev1, psi_rev1)
    rho_rev2 = reverse_qubit_order(rho)
    assert np.allclose(rho_rev1, rho_rev2)

    # draw n times 2 random 1-qubit states and a probability distribution over all n pairs
    n = 10
    psis = [[random_dm(1) for _ in range(2)] for _ in range(n)]
    p = normalize(np.random.rand(n), p=1)
    # compute the average state
    psi = np.zeros((2**2, 2**2), dtype=complex)
    for i in range(n):
        psi += p[i]*np.kron(psis[i][0], psis[i][1])
    # compute the average state with reversed qubit order
    psi_rev1 = np.zeros((2**2, 2**2), dtype=complex)
    for i in range(n):
        psi_rev1 += p[i]*np.kron(psis[i][1], psis[i][0])

    psi_rev2 = reverse_qubit_order(psi)
    assert np.allclose(psi_rev1, psi_rev2), f"psi_rev1 = {psi_rev1}\npsi_rev2 = {psi_rev2}"

def _test_partial_trace():
    # known 4x4 matrix
    rho = np.arange(16).reshape(4,4)
    rhoA_expect = np.array([[ 5, 9], [21, 25]])
    rhoA_actual = partial_trace(rho, 0)
    assert np.allclose(rhoA_actual, rhoA_expect), f"rho_actual = {rhoA_actual}\nrho_expect = {rhoA_expect}"
    rhoB_expect = np.array([[10, 12], [18, 20]])
    rhoB_actual = partial_trace(rho, [1], reorder=True)
    assert np.allclose(rhoB_actual, rhoB_expect), f"rho_actual = {rhoB_actual}\nrho_expect = {rhoB_expect}"

    # two separable density matrices
    rhoA = random_dm(2)
    rhoB = random_dm(3)
    rho = np.kron(rhoA, rhoB)
    rhoA_expect = rhoA
    rhoA_actual = partial_trace(rho, [0,1])
    assert np.allclose(rhoA_actual, rhoA_expect), f"rho_actual = {rhoA_actual}\nrho_expect = {rhoA_expect}"
    rhoA_actual = partial_trace(rho, [1,0], reorder=True)  # test order
    rhoA_expect = reverse_qubit_order(rhoA)
    assert np.allclose(rhoA_actual, rhoA_expect), f"rho_actual = {rhoA_actual}\nrho_expect = {rhoA_expect}"

    # two separable state vectors
    psiA = random_ket(2)
    psiB = random_ket(3)
    psi = np.kron(psiA, psiB)
    psiA_expect = np.outer(psiA, psiA.conj())
    psiA_actual = partial_trace(psi, [0,1])
    assert np.allclose(psiA_actual, psiA_expect), f"psi_actual = {psiA_actual}\npsi_expect = {psiA_expect}"

    # total trace
    st = random_ket(3)
    st_tr = partial_trace(st, [])
    assert np.allclose(st_tr, np.array([[1]])), f"st_tr = {st_tr} ≠ 1"
    rho = random_dm(3)
    rho_tr = partial_trace(rho, [])
    assert np.allclose(rho_tr, np.array([[1]])), f"rho_tr = {rho_tr} ≠ 1"

    # retain all qubits
    st = random_ket(3)
    st_tr = partial_trace(st, [0,1,2])
    st_expect = np.outer(st, st.conj())
    assert st_tr.shape == st_expect.shape, f"st_tr.shape = {st_tr.shape} ≠ st_expect.shape = {st_expect.shape}"
    assert np.allclose(st_tr, st_expect), f"st_tr = {st_tr} ≠ st_expect = {st_expect}"
    rho = random_dm(2)
    rho_tr = partial_trace(rho, [0,1])
    assert rho_tr.shape == rho.shape, f"rho_tr.shape = {rho_tr.shape} ≠ rho.shape = {rho.shape}"
    assert np.allclose(rho_tr, rho), f"rho_tr = {rho_tr} ≠ rho = {rho}"

    # batch dimension
    kets = random_ket(3, 10)
    rhos = partial_trace(kets, [0,1])
    assert np.allclose(np.trace(rhos, axis1=-2, axis2=-1), np.ones(10))
    rhos_rev  = partial_trace(kets, [1,0], reorder=True)
    rhos_rev2 = reverse_qubit_order(rhos, batch_shape=(10,))
    assert np.allclose(rhos_rev, rhos_rev2)

def _test_is_eigenstate():
    H = parse_hamiltonian('XX + YY + ZZ')
    assert is_eigenstate(ket('00'), H)
    assert not is_eigenstate(ket('01'), H)
    H = parse_hamiltonian('ZI - IZ')
    assert is_eigenstate(ket('10'), H)  # eigenvalue -2
    assert is_eigenstate(ket('01'), H)  # eigenvalue 2
    assert is_eigenstate(ket('00'), H)  # eigenvalue 0
    assert not is_eigenstate(ket('10 + 11'), H)

def _test_count_qubits():
    assert count_qubits(ising(20)) == 20
    assert count_qubits('CXC @ XCC') == 3
    assert count_qubits(parse_unitary('CXC @ XCC')) == 3
    assert count_qubits('0.001*01 - 0.101*11') == 2
    assert count_qubits(ket('0.001*01 - 0.101*11')) == 2
    qc = QuantumComputer(4)
    qc.x(4)
    assert count_qubits(qc) == 5

def _test_von_neumann_entropy():
    rho = random_dm(2, 'pure')
    S = von_neumann_entropy(rho)
    assert np.allclose(S, 0), f"S = {S} ≠ 0"

    rho = np.eye(2)/2
    S = von_neumann_entropy(rho)
    assert np.allclose(S, 1), f"S = {S} ≠ 1"

def _test_entanglement_entropy():
    # Two qubits in the Bell state |00> + |11> should have entropy 1
    rho = dm('00 + 11')
    S = entanglement_entropy(rho, [0])
    assert np.allclose(S, 1), f"S = {S} ≠ 1"

    # Two separable systems should have entropy 0
    rhoA = random_dm(2, 'pure')
    rhoB = random_dm(3, 'pure')
    rho = np.kron(rhoA, rhoB)
    S = entanglement_entropy(rho, [0,1])
    assert np.allclose(S, 0), f"S = {S} ≠ 0"

def _test_fidelity():
    # same state
    psi = random_ket(2)
    assert np.allclose(fidelity(psi, psi), 1), f"fidelity = {fidelity(psi, psi)} ≠ 1"
    rho = random_dm(2)
    assert np.allclose(fidelity(rho, rho), 1), f"fidelity = {fidelity(rho, rho)} ≠ 1"

    # orthogonal states
    psi1 = ket('00')
    psi2 = ket('11')
    assert np.allclose(fidelity(psi1, psi2), 0), f"fidelity = {fidelity(psi1, psi2)} ≠ 0"
    rho1 = dm('00')
    rho2 = dm('11')
    assert np.allclose(fidelity(rho1, rho2), 0), f"fidelity = {fidelity(rho1, rho2)} ≠ 0"
    assert np.allclose(fidelity(psi1, rho2), 0), f"fidelity = {fidelity(psi1, rho2)} ≠ 0"
    assert np.allclose(fidelity(rho1, psi2), 0), f"fidelity = {fidelity(rho1, psi2)} ≠ 0"

    # check values
    psi1 = ket('0')
    psi2 = ket('-')
    rho1 = dm('0')
    rho2 = dm('-')
    assert np.allclose(fidelity(psi1, psi2), 1/2), f"fidelity = {fidelity(psi1, psi2)} ≠ 1/2"
    assert np.allclose(fidelity(rho1, rho2), 1/2), f"fidelity = {fidelity(rho1, rho2)} ≠ 1/2"
    assert np.allclose(fidelity(psi1, rho2), 1/2), f"fidelity = {fidelity(psi1, rho2)} ≠ 1/2"
    assert np.allclose(fidelity(rho1, psi2), 1/2), f"fidelity = {fidelity(rho1, psi2)} ≠ 1/2"

    # random states to test properties: F(s1,s2) \in [0,1], symmetric
    psi1 = random_ket(2)
    psi2 = random_ket(2)
    assert 0 <= fidelity(psi1, psi2) <= 1, f"fidelity = {fidelity(psi1, psi2)} ∉ [0,1]"
    assert np.allclose(fidelity(psi1, psi2), fidelity(psi2, psi1)), f"fidelity(psi1, psi2) = {fidelity(psi1, psi2)} ≠ {fidelity(psi2, psi1)}"
    rho1 = random_dm(2)
    rho2 = random_dm(2)
    assert 0 <= fidelity(rho1, rho2) <= 1, f"fidelity = {fidelity(rho1, rho2)} ∉ [0,1]"
    assert np.allclose(fidelity(rho1, rho2), fidelity(rho2, rho1)), f"fidelity(rho1, rho2) = {fidelity(rho1, rho2)} ≠ {fidelity(rho2, rho1)}"
    assert np.allclose(fidelity(psi1, rho2), fidelity(rho2, psi1)), f"fidelity(psi1, rho2) = {fidelity(psi1, rho2)} ≠ {fidelity(rho2, psi1)}"
    assert np.allclose(fidelity(rho1, psi2), fidelity(psi2, rho1)), f"fidelity(rho1, psi2) = {fidelity(rho1, psi2)} ≠ {fidelity(psi2, rho1)}"
    assert 0 <= fidelity(psi1, rho2) <= 1, f"fidelity = {fidelity(psi1, rho2)} ∉ [0,1]"
    assert 0 <= fidelity(rho1, psi2) <= 1, f"fidelity = {fidelity(rho1, psi2)} ∉ [0,1]"

def _test_trace_distance():
    # same state
    psi = random_ket(2)
    assert np.isclose(trace_distance(psi, psi), 0), f"trace_distance = {trace_distance(psi, psi)} ≠ 0"
    rho = random_dm(2)
    assert np.isclose(trace_distance(rho, rho), 0), f"trace_distance = {trace_distance(rho, rho)} ≠ 0"
    # orthogonal states
    psi1, psi2 = '00', '11'
    assert np.isclose(trace_distance(psi1, psi2), 1), f"trace_distance = {trace_distance(psi1, psi2)} ≠ 1"
    # other values
    psi1, psi2 = '0', '-'
    assert np.isclose(trace_distance(psi1, psi2), fs(2)), f"trace_distance = {trace_distance(psi1, psi2)} ≠ 1/sqrt(2)"

def _test_schmidt_decomposition():
    n = 6
    subsystem = np.random.choice(n, size=np.random.randint(1, n), replace=False)
    subsystem = sorted(subsystem)  # TODO: figure out how to correctly transpose axes in partial trace with unsorted subsystems
    psi = random_ket(n)
    l, A, B = schmidt_decomposition(psi, subsystem)

    # check non-negativity of Schmidt coefficients
    assert np.all(l >= 0), f"l = {l} < 0"
    # check normalization of Schmidt coefficients
    assert np.allclose(np.sum(l**2), 1), f"sum(l**2) = {np.sum(l**2)} ≠ 1"

    # check RDM for subsystem A
    rho_expect = partial_trace(np.outer(psi, psi.conj()), subsystem)
    rho_actual = np.sum([l_i**2 * np.outer(A_i, A_i.conj()) for l_i, A_i in zip(l, A)], axis=0)
    assert np.allclose(rho_actual, rho_expect), f"rho_expect - rho_actual = {rho_expect - rho_actual}"

    # check RDM for subsystem B
    rho_expect = partial_trace(np.outer(psi, psi.conj()), [i for i in range(n) if i not in subsystem])
    rho_actual = np.sum([l_i**2 * np.outer(B_i, B_i.conj()) for l_i, B_i in zip(l, B)], axis=0)
    assert np.allclose(rho_actual, rho_expect), f"rho_expect - rho_actual = {rho_expect - rho_actual}"

    # check entanglement entropy
    S_expect = entanglement_entropy(psi, subsystem)
    S_actual = -np.sum([l_i**2 * np.log2(l_i**2) for l_i in l])
    assert np.allclose(S_actual, S_expect), f"S_expect = {S_expect} ≠ S_actual = {S_actual}"

def _test_purify():
    rho = random_dm(2, rank=2)
    psi = purify(rho)
    assert psi.shape == (8,), f"psi.shape = {psi.shape} ≠ (8,)"
    n = randint(2, 5)
    rho = random_dm(n, rank=randint(1, 2**n))
    psi = purify(rho)
    assert_ket(psi)
    assert np.allclose(partial_trace(psi, range(n)), rho)
    psi2 = purify(rho, ancilla_basis_cb=I_)
    assert_ket(psi2)
    assert np.allclose(partial_trace(psi2, range(n)), rho)

def _test_correlation_quantum():
    global XX, ZZ
    assert np.isclose(correlation_quantum(ket('0101 + 0000'), ZZ, ZZ), 1)
    assert np.isclose(correlation_quantum(ket('0101 + 0000'), XX, XX), 0)
    assert np.isclose(correlation_quantum(ket('0101 + 1010'), XX, XX), 1)
    assert np.isclose(correlation_quantum(ket('0101 + 1010'), ZZ, ZZ), 0)
    assert np.isclose(correlation_quantum(ket('0.5*0101 + 0000'), ZZ, ZZ), 0.64)
    assert np.isclose(correlation_quantum(dm('0.5*0101 + 0000'), ZZ, ZZ), 0.64)

def _test_ground_state():
    H = parse_hamiltonian('ZZII + IZZI + IIZZ', dtype=float)
    ge, gs = -3, ket('0101')

    res_exact = ground_state_exact(H)
    assert np.allclose(res_exact[0], ge)
    assert np.allclose(res_exact[1], gs)

    res_ITE = ground_state_ITE(H)
    assert np.allclose(res_ITE[0], ge)
    # assert np.allclose(res_ITE[1], gs) # might be complex due to random state initialization


def _test_ising():
    # 1d
    H_str = ising(5, J=1.5, h=0, g=0, offset=0, kind='1d', circular=False)
    expect = "1.5*(ZZIII + IZZII + IIZZI + IIIZZ)"
    assert np.allclose(ph(H_str), ph(expect)), f"\nH_str    = {H_str}\nexpect = {expect}"

    H_str = ising(3, J=0, h=3, g=2)
    expect = '3*(ZII + IZI + IIZ) + 2*(XII + IXI + IIX)'
    assert np.allclose(ph(H_str), ph(expect)), f"\nH_str  = {H_str}\nexpect = {expect}"

    H_str = ising(5, J=1.5, h=1.1, g=0.5, offset=0.5, kind='1d', circular=True)
    expect = "1.5*(ZZIII + IZZII + IIZZI + IIIZZ + ZIIIZ) + 1.1*(ZIIII + IZIII + IIZII + IIIZI + IIIIZ) + 0.5*(XIIII + IXIII + IIXII + IIIXI + IIIIX) + 0.5"
    assert np.allclose(ph(H_str), ph(expect)), f"\nH_str  = {H_str}\nexpect = {expect}"

    H_str = ising(3, J=[0.6,0.7,0.8], h=[0.1,0.2,0.7], g=[0.6,0.1,1.5], offset=0.5, kind='1d', circular=True)
    expect = "0.6*ZZI + 0.7*IZZ + 0.8*ZIZ + 0.1*ZII + 0.2*IZI + 0.7*IIZ + 0.6*XII + 0.1*IXI + 1.5*IIX + 0.5"
    assert np.allclose(ph(H_str), ph(expect)), f"\nH_str  = {H_str}\nexpect = {expect}"

    H_str = ising(3, J=[0,1], h=[1,2], g=[2,5], offset=0.5, kind='1d', circular=True)
    # random, but count terms in H_str instead
    n_terms = len(H_str.split('+'))
    assert n_terms == 10, f"n_terms = {n_terms}\nexpect = 10"

    # 2d
    H_str = ising((2,2), J=1.5, h=0, g=0, offset=0, kind='2d', circular=False)
    expect = "1.5*(ZIZI + ZZII + IZIZ + IIZZ)"
    assert H_str == expect, f"\nH_str  = {H_str}\nexpect = {expect}"

    H_str = ising((3,3), J=1.5, h=1.1, g=0.5, offset=0.5, kind='2d', circular=True)
    expect = "1.5*(ZIIZIIIII + ZZIIIIIII + IZIIZIIII + IZZIIIIII + IIZIIZIII + ZIZIIIIII + IIIZIIZII + IIIZZIIII + IIIIZIIZI + IIIIZZIII + IIIIIZIIZ + IIIZIZIII + IIIIIIZZI + ZIIIIIZII + IIIIIIIZZ + IZIIIIIZI + IIZIIIIIZ + IIIIIIZIZ) + 1.1*(ZIIIIIIII + IZIIIIIII + IIZIIIIII + IIIZIIIII + IIIIZIIII + IIIIIZIII + IIIIIIZII + IIIIIIIZI + IIIIIIIIZ) + 0.5*(XIIIIIIII + IXIIIIIII + IIXIIIIII + IIIXIIIII + IIIIXIIII + IIIIIXIII + IIIIIIXII + IIIIIIIXI + IIIIIIIIX) + 0.5"
    assert np.allclose(ph(H_str, sparse=True).data, ph(expect, sparse=True).data), f"\nH_str  = {H_str}\nexpect = {expect}"

    # 3d
    H_str = ising((2,2,3), kind='3d', J=1.2, h=0, g=0, offset=0, circular=True)
    expect = "1.2*(ZIIIIIZIIIII + ZIIZIIIIIIII + ZZIIIIIIIIII + IZIIIIIZIIII + IZIIZIIIIIII + IZZIIIIIIIII + IIZIIIIIZIII + IIZIIZIIIIII + ZIZIIIIIIIII + IIIZIIIIIZII + IIIZZIIIIIII + IIIIZIIIIIZI + IIIIZZIIIIII + IIIIIZIIIIIZ + IIIZIZIIIIII + IIIIIIZIIZII + IIIIIIZZIIII + IIIIIIIZIIZI + IIIIIIIZZIII + IIIIIIIIZIIZ + IIIIIIZIZIII + IIIIIIIIIZZI + IIIIIIIIIIZZ + IIIIIIIIIZIZ)"
    assert H_str == expect, f"\nH_str  = {H_str}\nexpect = {expect}"

    H_str = ising((3,3,3), kind='3d', J=1.5, h=0, g=0, offset=0, circular=True)
    expect = "1.5*(ZIIIIIIIIZIIIIIIIIIIIIIIIII + ZIIZIIIIIIIIIIIIIIIIIIIIIII + ZZIIIIIIIIIIIIIIIIIIIIIIIII + IZIIIIIIIIZIIIIIIIIIIIIIIII + IZIIZIIIIIIIIIIIIIIIIIIIIII + IZZIIIIIIIIIIIIIIIIIIIIIIII + IIZIIIIIIIIZIIIIIIIIIIIIIII + IIZIIZIIIIIIIIIIIIIIIIIIIII + ZIZIIIIIIIIIIIIIIIIIIIIIIII + IIIZIIIIIIIIZIIIIIIIIIIIIII + IIIZIIZIIIIIIIIIIIIIIIIIIII + IIIZZIIIIIIIIIIIIIIIIIIIIII + IIIIZIIIIIIIIZIIIIIIIIIIIII + IIIIZIIZIIIIIIIIIIIIIIIIIII + IIIIZZIIIIIIIIIIIIIIIIIIIII + IIIIIZIIIIIIIIZIIIIIIIIIIII + IIIIIZIIZIIIIIIIIIIIIIIIIII + IIIZIZIIIIIIIIIIIIIIIIIIIII + IIIIIIZIIIIIIIIZIIIIIIIIIII + IIIIIIZZIIIIIIIIIIIIIIIIIII + ZIIIIIZIIIIIIIIIIIIIIIIIIII + IIIIIIIZIIIIIIIIZIIIIIIIIII + IIIIIIIZZIIIIIIIIIIIIIIIIII + IZIIIIIZIIIIIIIIIIIIIIIIIII + IIIIIIIIZIIIIIIIIZIIIIIIIII + IIZIIIIIZIIIIIIIIIIIIIIIIII + IIIIIIZIZIIIIIIIIIIIIIIIIII + IIIIIIIIIZIIIIIIIIZIIIIIIII + IIIIIIIIIZIIZIIIIIIIIIIIIII + IIIIIIIIIZZIIIIIIIIIIIIIIII + IIIIIIIIIIZIIIIIIIIZIIIIIII + IIIIIIIIIIZIIZIIIIIIIIIIIII + IIIIIIIIIIZZIIIIIIIIIIIIIII + IIIIIIIIIIIZIIIIIIIIZIIIIII + IIIIIIIIIIIZIIZIIIIIIIIIIII + IIIIIIIIIZIZIIIIIIIIIIIIIII + IIIIIIIIIIIIZIIIIIIIIZIIIII + IIIIIIIIIIIIZIIZIIIIIIIIIII + IIIIIIIIIIIIZZIIIIIIIIIIIII + IIIIIIIIIIIIIZIIIIIIIIZIIII + IIIIIIIIIIIIIZIIZIIIIIIIIII + IIIIIIIIIIIIIZZIIIIIIIIIIII + IIIIIIIIIIIIIIZIIIIIIIIZIII + IIIIIIIIIIIIIIZIIZIIIIIIIII + IIIIIIIIIIIIZIZIIIIIIIIIIII + IIIIIIIIIIIIIIIZIIIIIIIIZII + IIIIIIIIIIIIIIIZZIIIIIIIIII + IIIIIIIIIZIIIIIZIIIIIIIIIII + IIIIIIIIIIIIIIIIZIIIIIIIIZI + IIIIIIIIIIIIIIIIZZIIIIIIIII + IIIIIIIIIIZIIIIIZIIIIIIIIII + IIIIIIIIIIIIIIIIIZIIIIIIIIZ + IIIIIIIIIIIZIIIIIZIIIIIIIII + IIIIIIIIIIIIIIIZIZIIIIIIIII + IIIIIIIIIIIIIIIIIIZIIZIIIII + IIIIIIIIIIIIIIIIIIZZIIIIIII + ZIIIIIIIIIIIIIIIIIZIIIIIIII + IIIIIIIIIIIIIIIIIIIZIIZIIII + IIIIIIIIIIIIIIIIIIIZZIIIIII + IZIIIIIIIIIIIIIIIIIZIIIIIII + IIIIIIIIIIIIIIIIIIIIZIIZIII + IIZIIIIIIIIIIIIIIIIIZIIIIII + IIIIIIIIIIIIIIIIIIZIZIIIIII + IIIIIIIIIIIIIIIIIIIIIZIIZII + IIIIIIIIIIIIIIIIIIIIIZZIIII + IIIZIIIIIIIIIIIIIIIIIZIIIII + IIIIIIIIIIIIIIIIIIIIIIZIIZI + IIIIIIIIIIIIIIIIIIIIIIZZIII + IIIIZIIIIIIIIIIIIIIIIIZIIII + IIIIIIIIIIIIIIIIIIIIIIIZIIZ + IIIIIZIIIIIIIIIIIIIIIIIZIII + IIIIIIIIIIIIIIIIIIIIIZIZIII + IIIIIIIIIIIIIIIIIIIIIIIIZZI + IIIIIIZIIIIIIIIIIIIIIIIIZII + IIIIIIIIIIIIIIIIIIZIIIIIZII + IIIIIIIIIIIIIIIIIIIIIIIIIZZ + IIIIIIIZIIIIIIIIIIIIIIIIIZI + IIIIIIIIIIIIIIIIIIIZIIIIIZI + IIIIIIIIZIIIIIIIIIIIIIIIIIZ + IIIIIIIIIIIIIIIIIIIIZIIIIIZ + IIIIIIIIIIIIIIIIIIIIIIIIZIZ)"
    assert H_str == expect, f"\nH_str  = {H_str}\nexpect = {expect}"

    # pairwise
    H_str = ising(4, J=-.5, h=.4, g=.7, offset=1, kind='pairwise')
    expect = "-0.5*(ZZII + ZIZI + ZIIZ + IZZI + IZIZ + IIZZ) + 0.4*(ZIII + IZII + IIZI + IIIZ) + 0.7*(XIII + IXII + IIXI + IIIX) + 1"
    assert np.allclose(ph(H_str), ph(expect)), f"\nH_str  = {H_str}\nexpect = {expect}"

    # full
    H_str = ising(3, J=1.5, h=.4, g=.7, offset=1, kind='all')
    expect = "1.5*(ZZI + ZIZ + IZZ + ZZZ) + 0.4*(ZII + IZI + IIZ) + 0.7*(XII + IXI + IIX) + 1"
    assert np.allclose(ph(H_str), ph(expect)), f"\nH_str  = {H_str}\nexpect = {expect}"

    H_str = ising(3, kind='all', J={(0,1): 2, (0,1,2): 3, (1,2):0}, g=0, h=1.35)
    expect = "2*ZZI + 3*ZZZ + 1.35*(ZII + IZI + IIZ)"
    assert np.allclose(ph(H_str), ph(expect)), f"\nH_str  = {H_str}\nexpect = {expect}"

    J_dict = {
        (0,1): 1.5,
        (0,2): 2,
        (1,2): 0.5,
        (0,1,2): 3,
        (0,1,2,3): 0.5
    }
    H_str = ising(4, J=J_dict, h=.3, g=.5, offset=1.2, kind='all')
    expect = "1.5*ZZII + 2*ZIZI + 0.5*IZZI + 3*ZZZI + 0.5*ZZZZ + 0.3*(ZIII + IZII + IIZI + IIIZ) + 0.5*(XIII + IXII + IIXI + IIIX) + 1.2"
    assert np.allclose(ph(H_str), ph(expect)), f"\nH_str  = {H_str}\nexpect = {expect}"

def _test_pauli_basis():
    n = np.random.randint(1,4)
    pauli_n = list(pauli_basis(n))

    # check the number of generators
    n_expect = 2**(2*n)
    assert len(pauli_n) == n_expect, f"Number of generators is {len(pauli_n)}, but should be {n_expect}!"

    # check if no two generators are the same
    for i, (A,B) in enumerate(itertools.combinations(pauli_n,2)):
        assert not np.allclose(A, B), f"Pair {i} (n = {n}) is not different!"

    # check if all generators except of the identity are traceless
    assert np.allclose(pauli_n[0], np.eye(2**n)), "First generator is not the identity!"
    for i, A in enumerate(pauli_n[1:]):
        assert np.isclose(np.trace(A), 0), f"Generator {i} (n = {n}) is not traceless!"

    # check if all generators are Hermitian
    for i, A in enumerate(pauli_n):
        assert is_hermitian(A), f"Generator {i} (n = {n}) is not Hermitian!"

    # check if all generators are orthogonal
    for i, (A,B) in enumerate(itertools.combinations(pauli_n,2)):
        assert np.allclose(np.trace(A.conj().T @ B), 0), f"Pair {i} (n = {n}) is not orthogonal!"

    # check normalization
    pauli_n_norm = pauli_basis(n, kind='np', normalize=True)
    for i, A in enumerate(pauli_n_norm):
        assert np.isclose(np.linalg.norm(A), 1), f"Generator {i} (n = {n}) does not have norm 1! {np.linalg.norm(A)}"

    # check string representation
    pauli_n_str = list(pauli_basis(n, kind='str'))
    assert len(pauli_n) == len(pauli_n_str), "Number of generators is not the same!"

    # check if all generators are the same
    for i, (A,B) in enumerate(zip(pauli_n, pauli_n_str)):
        assert np.allclose(A, parse_hamiltonian(B)), f"Generator {i} (n = {n}) is not the same! {B}\n{A}\n≠\n{parse_hamiltonian(B)}"

    # check sparse representation
    pauli_n_sp = list(pauli_basis(n, kind='sp'))
    assert len(pauli_n) == len(pauli_n_sp), "Number of generators is not the same!"

    # check if all generators are the same
    for i, (A,B) in enumerate(zip(pauli_n, pauli_n_sp)):
        assert np.allclose(A, B.todense()), f"Generator {i} (n = {n}) is not the same!"

def _test_pauli_decompose():
    # H_gate = (X+Z)/sqrt(2)
    coeff, basis = pauli_decompose(H_gate)
    assert np.allclose(coeff, [fs2]*2), f"coeff = {coeff} ≠ [{fs2}]*2"
    assert basis == ['X', 'Z'], f"basis = {basis} ≠ ['X', 'Z']"

    # SWAP = 0.5*(II + XX + YY + ZZ)
    coeff, basis = pauli_decompose(SWAP)
    assert np.allclose(coeff, [0.5]*4), f"coeff = {coeff} ≠ [0.5]*4"
    assert basis == ['II', 'XX', 'YY', 'ZZ'], f"basis = {basis} ≠ ['II', 'XX', 'YY', 'ZZ']"

    # random n-qubit hamiltonian
    N = 2**randint(1, 5)
    H = random_hermitian(N)
    coeff, basis = pauli_decompose(H, eps=1e-10)
    assert len(coeff) == N**2, f"len(coeff) = {len(coeff)} ≠ {N**2}"
    assert len(basis) == N**2, f"len(basis) = {len(basis)} ≠ {N**2}"

    # check if the decomposition is correct
    H_recomposed = sum([c*parse_hamiltonian(b) for c, b in zip(coeff, basis)])
    assert np.allclose(H, H_recomposed), f"H = {H}\nH_decomposed = {H_recomposed}"
    H_str = pauli_decompose(H, eps=1e-10, as_str=True)
    H_recomposed2 = parse_hamiltonian(H_str)
    assert np.allclose(H, H_recomposed2, atol=1e-9), f"H = {H}\nH_decomposed = {H_recomposed2}"

    # check if `eps=0` always returns the whole basis
    n = 4
    coeff, basis = pauli_decompose(np.eye(2**n), eps=0)
    n_expect = 2**(2*n)  # == len(pauli_basis(n))
    assert len(coeff) == n_expect, f"len(coeff) = {len(coeff)} ≠ {n_expect}"
    assert len(basis) == n_expect, f"len(basis) = {len(basis)} ≠ {n_expect}"
