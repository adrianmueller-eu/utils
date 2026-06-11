This submodule contains a wide range of functionalities for quantum computing.

Let's create a simple |01⟩ state and apply a SWAP gate:

```python
from utils.quantum import *

psi = ket('01')
SWAP = parse_unitary('CX @ XC @ CX')
unket(SWAP @ psi)
```
```
'10'
```

For interactive simulation — circuits, measurements, noise, operator tracking, phase estimation, QFT — use the standalone [`QuantumComputer`](https://github.com/noxafy/QuantumComputer) package.

States can be expressed as strings, integers, or arrays. Density matrices are supported throughout.

```python
psi = ket('00 + 11')          # Bell state
rho = dm('00 + 11')           # same, as density matrix
unket(psi)                    # → '00 + 11'
is_ket(psi)                   # → True
psi = random_ket(4)           # random Haar-distributed state
rho = random_dm(4, rank=2)    # random mixed state of given rank
```

Build unitaries and Hamiltonians from strings — controls, tensor products, parentheses, the works.

```python
parse_unitary('CX @ XC @ CX')       # SWAP
parse_unitary('CCX')                # Toffoli
parse_unitary('CXC')                # control on qubits 0 and 2
parse_unitary('NXC')                # negative control on qubit 0

parse_hamiltonian('XX + ZZ')
parse_hamiltonian('0.5*(XX + YY + ZZ + II)')  # this is SWAP
```

Generate random Ising models and find their ground states:

```python
H = ising((2,3), kind='2d', circular=True)  # 2×3 lattice, periodic boundaries
H = parse_hamiltonian(H)
energy, ground_state = ground_state_exact(H)
print(f"Ground state energy: {energy}")

psi = random_ket(count_qubits(H))
print(f"Energy of random state: {ev(H, psi)}")
```
```
Ground state energy: -6.1492550068479614
Energy of random state: -0.8639892573384738
```

Quantum information metrics work on both kets and density matrices:

```python
S = von_neumann_entropy(rho)              # von Neumann entropy
S = entanglement_entropy(psi, [0,1])       # of a subsystem
F = fidelity(psi, rho)                    # state fidelity
l, A, B = schmidt_decomposition(psi, [0,1])  # Schmidt coefficients & vectors
p = purity(rho)
coeffs, basis = pauli_decompose(H)        # decompose into Pauli strings
```

Enjoy! ❤️