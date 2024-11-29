This submodule contains a wide range of functionalities for quantum computing.

Here are some examples:
```python
psi = ket('01')
SWAP = parse_unitary('CX @ XC @ CX')
print(unket(swap @ psi))
```
```
'10'
```

This creates a [SWAP gate](https://en.wikipedia.org/wiki/Quantum_logic_gate#Swap_gate) and swaps `01` to `10`. Alternatively, we could have created the matrix using `parse_hamiltonian('0.5*(XX + YY + ZZ + II)')`. The following example uses `parse_hamiltonian` in combination with `ising` to generate an ising hamiltonian
```python
H = parse_hamiltonian(ising((2,3), kind='2d', circular=True))  # random ising model on a 2x3 lattice with periodic boundary conditions
ge, gs = ground_state_exact(H)
print("Ground state energy:", ge)
n = count_qubits(H)
psi = random_ket(n)
print("Energy of random state:", ev(H, psi))
```
```
Ground state energy: -6.1492550068479614
Energy of random state: -0.8639892573384738
```

The class `QuantumComputer` allows to do interesting things. For example, the last part above could have been written as `QC(n, 'random').ev(H)`. Let's create a simple Bell state
```python
qc = QuantumComputer(2)
qc.h(0).cx(0, 1)  # create a Bell state
print(qc)  # show the current state
```
```
qubits [0, 1] in state '0.70711*(00+11)'
```

Continuing this example, we find
```python
print("Unitary is\n", qc.get_U())  # the generated unitary should be the same as parse_unitary('CX @ HI')
print("Subsystem of the first qubit\n", qc[0])  # Maximally mixed state
print("Entropy of a Bell state in XX:", entropy(qc.probs(obs=XX)))  # show the entropy of the Bell state in the XX basis

outcome = qc.measure()  # measure the state in the standard basis
print("Measurement result:", outcome)
```
```
Unitary is
 [[ 0.707+0.j  0.   +0.j  0.707+0.j  0.   +0.j]
 [ 0.   +0.j  0.707+0.j  0.   +0.j  0.707+0.j]
 [ 0.   +0.j  0.707+0.j  0.   +0.j -0.707+0.j]
 [ 0.707+0.j  0.   +0.j -0.707+0.j  0.   +0.j]]
Subsystem of the first qubit
[[0.5+0.j 0. +0.j]
[0. +0.j 0.5+0.j]]
Entropy of a Bell state in XX: 0.0
Measurement result: 11
```

The class allows to give qubits names as well as to dynamically add and remove them.
```python
qc = QuantumComputer(list('abc'), '000 + 111')  # create a GHZ state
print(qc)  # show the current state
qc.remove('b', collapse=True)  # remove the last qubits
print(qc)  # show the current state again
qc['a'] = random_dm(1)  # re-initialize the first qubit to a random density matrix
print(qc)  # look at the state now
```
```
qubits ['a', 'b', 'c'] in state '0.70711*(000+111)'
qubits ['a', 'c'] in state '11'
qubits ['a', 'c'] in state
[[0.     +0.j     0.     +0.j     0.     +0.j     0.     +0.j    ]
 [0.     +0.j     0.66292+0.j     0.     +0.j     0.20887-0.3755j]
 [0.     +0.j     0.     +0.j     0.     +0.j     0.     +0.j    ]
 [0.     +0.j     0.20887+0.3755j 0.     +0.j     0.33708-0.j    ]]
```

Here is a slightly more complex example
```python
qc = QuantumComputer(6)  # create a quantum simulator initialized with a 8-qubit random mixed state
qc[3:5] = random_dm(2)  # initialize qubits [2,3] to a random density matrix
qc.qft(list(range(6)))  # apply the quantum Fourier transform to qubits [0,1,2,3]
qc.decohere([2,4])  # decohere qubits [2,4]
qc.entanglement_entropy_pp('desc', head=5)  # show the remaining entanglement between all qubit bipartitions
# qc.plot()  # try this, also try calling qc.purify(True) before plotting
# qc.plotU() # comment out qc.decohere above and then try this
```
```
Top 5 bipartitions:
---------------
0 1 2 3 4  |  5 	3.0805158
1 2 3 4  |  0 5 	3.0805158
1 2 3 4 5  |  0 	3.0685178
0 1 2 4 5  |  3 	2.8515369
1 2 4 5  |  0 3 	2.8515369
```