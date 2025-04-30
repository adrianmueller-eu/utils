This submodule contains a wide range of functionalities for quantum computing.

Let create a simple |01> state and apply a SWAP gate
```py
from utils.quantum import *

psi = ket('01')
SWAP = parse_unitary('CX @ XC @ CX')
unket(SWAP @ psi)
```
```
'10'
```
The class [`QuantumComputer`](/quantum/QuantumComputer.py) allows to do interesting things. The same as above may look like
```py
from utils.quantum import QuantumComputer as QC

qc = QC('01')
qc(SWAP, [0,1])
```
```
qubits (0, 1) in state '10'
```
It allows to track the total quantum channel of the operations that have been applied
```py
qc = QC()         # empty instance
qc.h(0).cx(0, 1)  # create a Bell state
maximally_mixed_state = np.eye(2)/2
assert np.allclose(qc[0], maximally_mixed_state)  # use indexing / slicing to obtain the respective reduced density matrix
S = entropy(qc.probs(obs=XX))
print(f"Entropy of a Bell state under the XX observable: {S:.7f}")
qc.get_unitary()  # same as parse_unitary('CX @ HI')
```
```
Entropy of a Bell state under the XX observable: -0.0000000

array([[ 0.70710678+0.j,  0.        +0.j,  0.70710678+0.j,  0.        +0.j],
       [ 0.        +0.j,  0.70710678+0.j,  0.        +0.j,  0.70710678+0.j],
       [ 0.        +0.j,  0.70710678+0.j,  0.        +0.j, -0.70710678+0.j],
       [ 0.70710678+0.j,  0.        +0.j, -0.70710678+0.j,  0.        +0.j]])
```
If we measure "without looking" (decoherence), we break unitarity and thus switch into density matrix mode
```py
qc.decohere()
print("Number of Kraus operators:", len(qc.get_operators()))
qc
```
```
Number of Kraus operators: 4

qubits (0, 1) in state
[[0.5+0.j 0. +0.j 0. +0.j 0. +0.j]
 [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
 [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
 [0. +0.j 0. +0.j 0. +0.j 0.5+0.j]]
```
And of course we can measure
```python
qc.measure()  # measure the state in the standard basis
```
```
'11'
```
Here is a little bit more involved example using phase estimation
```py
θ = np.pi/4
U = np.kron(Rz(θ), Rz(θ))
state_reg  = range(2)
energy_reg = ['e0', 'e1', 'e2']  # we also can use strings as qubit identifiers

psi0 = ket('00+11')
qc = QC(state_reg, psi0)
qc.add(energy_reg)                      # add the energy register
qc.pe(U, state_reg, energy_reg)         # apply phase estimation
qc.measure(energy_reg, collapse=False)  # measure the energy register without collapse
qc.remove(energy_reg)                   # remove the energy register again
ops = qc.get_operators()

# verify the generated operators
assert_kraus(ops)
qc2 = QC(psi0)(ops)  # apply generated quantum channel to a fresh instance
np.allclose(qc2[:], qc[:])
```
```
True
```

The following example uses `ising` in combination with `parse_hamiltonian` to generate a random [transverse Ising hamiltonian](https://en.wikipedia.org/wiki/Transverse-field_Ising_model)
```py
H_str = ising((2,3), kind='2d', circular=True)  # random ising model on a 2x3 lattice with periodic boundary conditions
H = parse_hamiltonian(H_str)
ge, gs = ground_state_exact(H)  # use diagonalization to find the ground state
print("Ground state energy:", ge)
n = count_qubits(H)
psi = random_ket(n)
print("Energy of random state:", ev(H, psi))
```
```
Ground state energy: -6.1492550068479614
Energy of random state: -0.8639892573384738
```

Enjoy! ❤️