from .constants import *
from .state import *
from .hamiltonian import *
from .unitary import *
from .quantum_info import *
from .QuantumComputer import *

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
from ..mathlib import is_involutory, anticommute, float_from_binstr, random_hermitian

def test_quantum_all():
    tests = [
        _test_constants,
        _test_fourier_matrix,
        _test_parse_unitary,
        _test_parse_hamiltonian,  # required by _test_random_ham
        _test_random_ham,  # required by _test_exp_i
        _test_exp_i,
        _test_get_H_energies_eq_get_pe_energies,
        _test_QuantumComputer,
        _test_random_ket,  # required _test_reverse_qubit_order
        _test_random_dm,   # required by _test_partial_trace
        _test_reverse_qubit_order,
        _test_partial_trace,
        _test_ket_unket,
        _test_op_dm,
        _test_is_dm,
        _test_is_eigenstate,
        _test_count_qubits,
        _test_entropy_von_Neumann,
        _test_entropy_entanglement,
        _test_fidelity,
        _test_trace_distance,
        _test_Schmidt_decomposition,
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

def _test_fourier_matrix():
    assert np.allclose(Fourier_matrix(1), H)
    n = randint(2,8)
    F = Fourier_matrix(n)
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
    U_actual = transpose_qubit_order(parse_unitary('IXZ'), [0,2,1])
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
    assert np.allclose(U_expect, U_actual), f"H_str: {H_str}"
    U_actual = exp_i(H).get_unitary()
    assert np.allclose(U_expect, U_actual), f"H_str: {H_str}"

    U_expect = matexp(1j*3*H)
    U_actual = get_unitary(exp_i(H)**3)
    assert np.allclose(U_expect, U_actual), f"H_str: {H_str}"
    U_actual = (exp_i(H)**3).get_unitary()
    assert np.allclose(U_expect, U_actual), f"H_str: {H_str}"
    U_actual = exp_i(H).get_unitary(3)
    assert np.allclose(U_expect, U_actual), f"H_str: {H_str}"
    U_actual = exp_i(H, k=3).get_unitary()
    assert np.allclose(U_expect, U_actual), f"H_str: {H_str}"

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
    max_time = 0.02
    start = time.time()
    for _ in range(10):
        assert time.time() - start < max_time, f"ket is too slow (iteration {_}/10)"
        for _ in range(100):
            ket(psi)

def _test_random_ket():
    n_qubits = np.random.randint(1, 10)
    psi = random_ket(n_qubits)
    assert psi.shape == (2**n_qubits,)
    assert np.allclose(np.linalg.norm(psi), 1)

def _test_op_dm():
    assert np.allclose(op(0),   [[1,0], [0,0]])
    assert np.allclose(op(0,1), [[0,1], [0,0]])
    assert np.allclose(op(1,0), [[0,0], [1,0]])
    assert np.allclose(op(1),   [[0,0], [0,1]])
    O = op(0,3,n=2)
    assert O.shape[0] == O.shape[1]
    # op should be fast enough to return already-density-matrices 1000 times in negligible time
    import time
    rho = random_dm(2)
    max_time = 0.01
    start = time.time()
    for _ in range(10):
        assert time.time() - start < max_time, f"op is too slow (iteration {_}/10)"
        for _ in range(100):
            op(rho)

def _test_is_dm():
    assert is_dm(np.eye(2**2)/2**2)
    # random Bloch vectors
    v = np.random.uniform(-1, 1, 3)
    if np.linalg.norm(v) > 1:
        v = normalize(v)
    # create dm from Bloch vector
    rho = (I + v[0]*X + v[1]*Y + v[2]*Z)/2
    assert is_dm(rho)

def _test_random_dm():
    n_qubits = np.random.randint(1, 5)
    rho = random_dm(n_qubits)
    assert is_dm(rho)

def _test_QuantumComputer():
    # test basic functionality
    qc = QuantumComputer()
    qc.x(0)
    assert unket(qc.get_state()) == '1'
    qc.cx(0, 1)
    assert unket(qc.get_state()) == '11'
    qc.reset()
    assert unket(qc.get_state()) == '00'
    qc.h()
    qc.x(2)
    qc.init(2, [0,1])
    assert unket(qc.get_state()) == '101'
    qc.z([0,2])  # noop
    qc.swap(1,2)
    assert unket(qc.get_state()) == '110'
    qc.reset(1)
    assert unket(qc.get_state()) == '100'
    qc.remove([0,2])
    assert unket(qc.get_state()) == '0'
    qc.x(2)
    result = qc.measure('all')
    assert result == '01'

    # Heisenberg uncertainty principle
    qc = QuantumComputer(1, 'random')
    assert qc.std(X) * qc.std(Z) >= abs(qc.ev(1j*(X@Z - Z@X)))/2

    # Bell basis
    Bell = [
        ket('00 + 11'),
        ket('00 - 11'),
        ket('01 + 10'),
        ket('01 - 10')
    ]
    qc = QuantumComputer(2)
    qc.h(0)
    qc.cx(0, 1)
    qc(I, 0)  # let it reshape to [2,2]
    assert np.allclose(qc[0], I/2)
    assert np.allclose(qc[1], I/2)
    p = qc.probs(obs=XX)  # Bell basis is the basis of XX / YY observable -> deterministic outcome
    assert np.isclose(entropy(p), 0), f"p = {p}"
    with qc.observable(YY):  # test context manager
        p = qc.probs()
        assert np.isclose(entropy(p), 0), f"p = {p}"
    U_expected = parse_unitary('CX @ HI')  # check generated unitary
    assert np.allclose(qc.get_U(), U_expected), f"Incorrect unitary:\n{qc.U}\n ≠\n{U_expected}"

    assert np.isclose(qc.std(X, 0), 1)
    assert np.isclose(qc.std(Y, 0), 1)
    assert np.isclose(qc.std(Z, 0), 1)
    assert np.isclose(qc.ev(X, 0), 0)
    assert np.isclose(qc.ev(Y, 0), 0)
    assert np.isclose(qc.ev(Z, 0), 0)

    assert np.allclose(qc.schmidt_coefficients([0]), [.5, .5])
    qc = QuantumComputer(2, '00 + 01')
    assert np.allclose(qc.schmidt_coefficients([1]), [1])

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
    assert np.isclose(np.trace(H @ op('00')), float_from_binstr('.011'))  # 00 is eigenstate with energy 0.375 = '011'
    state_qubits = ['s0', 's1']
    qc = QuantumComputer(state_qubits)
    E_qubits = ['e0', 'e1', 'e2']
    qc.pe(U, state_qubits, E_qubits)
    res = qc.measure(E_qubits)
    assert res == '011', f"measurement result was {res} ≠ '011'"

    qc.remove('s0')
    assert np.allclose(qc.get_state(), ket('0011'))
    qc.remove('all')

    # test schmidt decomposition
    qc = QuantumComputer(5, 'random')
    bip = choice([i for i,o in bipartitions(range(5))])
    U, S, V = qc.schmidt_decomposition(bip)
    assert np.allclose(U @ np.diag(S) @ V.T.conj(), qc.get_state(bip))

    # test density matrix
    qc = QuantumComputer(2, '00 + 11')
    assert np.allclose(qc.get_state(), normalize([1,0,0,1]))
    qc.decohere()
    assert qc.is_matrix_mode()
    assert np.allclose(qc.get_state(), (op('00') + op('11'))/2)
    qc.purify()
    assert not qc.is_matrix_mode()
    qc.remove([0])

    qc = QuantumComputer(4, '0100 + 1010')
    assert np.isclose(qc.entanglement_entropy(3), 0)
    qc.remove(3)
    assert not qc.is_matrix_mode()
    assert np.isclose(qc.entanglement_entropy(1), 1)
    qc.remove(1)
    assert qc.is_matrix_mode()
    qc.x(2)
    assert np.allclose(qc.get_state(), (op('01') + op('10'))/2)
    qc.purify(sample=True)

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
    assert np.allclose(rhoA_expect, rhoA_actual), f"rho_expect = {rhoA_expect}\nrho_actual = {rhoA_actual}"

    # two separable density matrices
    rhoA = random_dm(2)
    rhoB = random_dm(3)
    rho = np.kron(rhoA, rhoB)
    rhoA_expect = rhoA
    rhoA_actual = partial_trace(rho, [0,1])
    assert np.allclose(rhoA_expect, rhoA_actual), f"rho_expect = {rhoA_expect}\nrho_actual = {rhoA_actual}"

    # two separable state vectors
    psiA = random_ket(2)
    psiB = random_ket(3)
    psi = np.kron(psiA, psiB)
    psiA_expect = np.outer(psiA, psiA.conj())
    psiA_actual = partial_trace(psi, [0,1])
    assert np.allclose(psiA_expect, psiA_actual), f"psi_expect = {psiA_expect}\npsi_actual = {psiA_actual}"

    # total trace
    st = random_ket(3)
    st_tr = partial_trace(st, [])
    assert np.allclose(np.array([[1]]), st_tr), f"st_tr = {st_tr} ≠ 1"
    rho = random_dm(3)
    rho_tr = partial_trace(rho, [])
    assert np.allclose(np.array([[1]]), rho_tr), f"rho_expect = {rhoA_expect}\nrho_actual = {rhoA_actual}"

    # retain all qubits
    st = random_ket(3)
    st_tr = partial_trace(st, [0,1,2])
    st_expect = np.outer(st, st.conj())
    assert st_expect.shape == st_tr.shape, f"st_expect.shape = {st_expect.shape} ≠ st_tr.shape = {st_tr.shape}"
    assert np.allclose(st_expect, st_tr), f"st_expect = {st_expect} ≠ st_tr = {st_tr}"
    rho = random_dm(2)
    rho_tr = partial_trace(rho, [0,1])
    assert rho.shape == rho_tr.shape, f"rho.shape = {rho.shape} ≠ rho_tr.shape = {rho_tr.shape}"
    assert np.allclose(rho, rho_tr), f"rho_expect = {rhoA_expect}\nrho_actual = {rhoA_actual}"

def _test_is_eigenstate():
    H = parse_hamiltonian('XX + YY + ZZ')
    assert is_eigenstate(ket('00'), H)
    assert not is_eigenstate(ket('01'), H)

def _test_count_qubits():
    assert count_qubits(ising(20)) == 20
    assert count_qubits('CXC @ XCC') == 3
    assert count_qubits(parse_unitary('CXC @ XCC')) == 3
    assert count_qubits('0.001*01 - 0.101*11') == 2
    assert count_qubits(ket('0.001*01 - 0.101*11')) == 2
    qc = QuantumComputer(4)
    qc.x(4)
    assert count_qubits(qc) == 5

def _test_entropy_von_Neumann():
    rho = random_dm(2, pure=True)
    S = entropy_von_Neumann(rho)
    assert np.allclose(S, 0), f"S = {S} ≠ 0"

    rho = np.eye(2)/2
    S = entropy_von_Neumann(rho)
    assert np.allclose(S, 1), f"S = {S} ≠ 1"

def _test_entropy_entanglement():
    # Two qubits in the Bell state |00> + |11> should have entropy 1
    rho = op('00 + 11')
    S = entropy_entanglement(rho, [0])
    assert np.allclose(S, 1), f"S = {S} ≠ 1"

    # Two separable systems should have entropy 0
    rhoA = random_dm(2, pure=True)
    rhoB = random_dm(3, pure=True)
    rho = np.kron(rhoA, rhoB)
    S = entropy_entanglement(rho, [0,1])
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

def _test_Schmidt_decomposition():
    n = 6
    subsystem = np.random.choice(n, size=np.random.randint(1, n), replace=False)
    subsystem = sorted(subsystem)  # TODO: figure out how to correctly transpose axes in partial trace with unsorted subsystems
    psi = random_ket(n)
    l, A, B = Schmidt_decomposition(psi, subsystem)

    # check non-negativity of Schmidt coefficients
    assert np.all(l >= 0), f"l = {l} < 0"
    # check normalization of Schmidt coefficients
    assert np.allclose(np.sum(l**2), 1), f"sum(l**2) = {np.sum(l**2)} ≠ 1"

    # check RDM for subsystem A
    rho_expect = partial_trace(np.outer(psi, psi.conj()), subsystem)
    rho_actual = np.sum([l_i**2 * np.outer(A_i, A_i.conj()) for l_i, A_i in zip(l, A)], axis=0)
    assert np.allclose(rho_expect, rho_actual), f"rho_expect - rho_actual = {rho_expect - rho_actual}"

    # check RDM for subsystem B
    rho_expect = partial_trace(np.outer(psi, psi.conj()), [i for i in range(n) if i not in subsystem])
    rho_actual = np.sum([l_i**2 * np.outer(B_i, B_i.conj()) for l_i, B_i in zip(l, B)], axis=0)
    assert np.allclose(rho_expect, rho_actual), f"rho_expect - rho_actual = {rho_expect - rho_actual}"

    # check entanglement entropy
    S_expect = entropy_entanglement(psi, subsystem)
    S_actual = -np.sum([l_i**2 * np.log2(l_i**2) for l_i in l])
    assert np.allclose(S_expect, S_actual), f"S_expect = {S_expect} ≠ S_actual = {S_actual}"

def _test_correlation_quantum():
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
    assert np.allclose(ph(H_str, sparse=True), ph(expect, sparse=True)), f"\nH_str  = {H_str}\nexpect = {expect}"

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
    pauli_n = pauli_basis(n)

    # check the number of generators
    n_expect = 2**(2*n)
    assert len(pauli_n) == n_expect, f"Number of generators is {len(pauli_n)}, but should be {n_expect}!"

    # check if no two generators are the same
    for i, (A,B) in enumerate(itertools.combinations(pauli_n,2)):
        assert not np.allclose(A, B), f"Pair {i} is not different!"

    # check if all generators except of the identity are traceless
    assert np.allclose(pauli_n[0], np.eye(2**n)), "First generator is not the identity!"
    for i, A in enumerate(pauli_n[1:]):
        assert np.isclose(np.trace(A), 0), f"Generator {i} is not traceless!"

    # check if all generators are Hermitian
    for i, A in enumerate(pauli_n):
        assert is_hermitian(A), f"Generator {i} is not Hermitian!"

    # check if all generators are orthogonal
    for i, (A,B) in enumerate(itertools.combinations(pauli_n,2)):
        assert np.allclose(np.trace(A.conj().T @ B), 0), f"Pair {i} is not orthogonal!"

    # check normalization
    pauli_n_norm = pauli_basis(n, kind='np', normalize=True)
    for i, A in enumerate(pauli_n_norm):
        assert np.isclose(np.linalg.norm(A), 1), f"Generator {i} does not have norm 1!"

    # check string representation
    pauli_n_str = pauli_basis(n, kind='str')
    assert len(pauli_n) == len(pauli_n_str), "Number of generators is not the same!"

    # check if all generators are the same
    for i, (A,B) in enumerate(zip(pauli_n, pauli_n_str)):
        assert np.allclose(A, parse_hamiltonian(B)), f"Generator {i} is not the same!"

    # check sparse representation
    pauli_n_sp = pauli_basis(n, kind='sp')
    assert len(pauli_n) == len(pauli_n_sp), "Number of generators is not the same!"

    # check if all generators are the same
    for i, (A,B) in enumerate(zip(pauli_n, pauli_n_sp)):
        assert np.allclose(A, B.todense()), f"Generator {i} is not the same!"

def _test_pauli_decompose():
    global f2, H_gate, SWAP
    H = H_gate

    # H = (X+Z)/sqrt(2)
    coeff, basis = pauli_decompose(H)
    assert np.allclose(coeff, [f2]*2), f"coeff = {coeff} ≠ [{f2}]*2"
    assert basis == ['X', 'Z'], f"basis = {basis} ≠ ['X', 'Z']"

    # SWAP = 0.5*(II + XX + YY + ZZ)
    coeff, basis = pauli_decompose(SWAP)
    assert np.allclose(coeff, [0.5]*4), f"coeff = {coeff} ≠ [0.5]*4"
    assert basis == ['II', 'XX', 'YY', 'ZZ'], f"basis = {basis} ≠ ['II', 'XX', 'YY', 'ZZ']"

    # random 2-qubit hamiltonian
    H = random_hermitian(4)
    coeff, basis = pauli_decompose(H)
    assert len(coeff) == 16, f"len(coeff) = {len(coeff)} ≠ 16"
    assert len(basis) == 16, f"len(basis) = {len(basis)} ≠ 16"

    # check if the decomposition is correct
    H_decomposed = np.zeros((4,4), dtype=complex)
    for c, b in zip(coeff, basis):
        H_decomposed += c*parse_hamiltonian(b)
    assert np.allclose(H, H_decomposed), f"H = {H}\nH_decomposed = {H_decomposed}"

    # check if `include_zero` returns the whole basis
    n = 4
    coeff, basis = pauli_decompose(np.eye(2**n), eps=0)
    n_expect = 2**(2*n)  # == len(pauli_basis(n))
    assert len(coeff) == n_expect, f"len(coeff) = {len(coeff)} ≠ {n_expect}"
    assert len(basis) == n_expect, f"len(basis) = {len(basis)} ≠ {n_expect}"