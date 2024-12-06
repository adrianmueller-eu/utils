### Check levels

Let `A` be a matrix of `size = 2**10`, `dtype = complex`, and appropriate type (e.g. hermitian).
Let `L` be a vector of `size = 2**10` and `dtype = complex`.

Level 0: Virtually instant, no restriction
| function | time (ms) |
--|--
| A.T | 0.00005
| A.ravel | 0.00005
| A.real | 0.00005
| A.reshape([2]*20) | 0.0004
| L.conj() | 0.0005
| np.diag(A) | 0.0006
| L @ L | 0.0008
| np.trace | 0.0013
| L.conj() @ L | 0.0014
| np.sum(L) | 0.0016
| L/1 | 0.0019
| np.all(L > 0) | 0.0025
| is_ket(L) | 0.003
| norm(L) | 0.0048
| normalize(L) | 0.0052
| np.allclose(L, L) | 0.015

Level 1: Minimal burden, disable only for high performance
| function | time (ms) |
--|--
| np.all(A.real > 0) | 0.45
| A.conj() | 0.6
| is_eigenstate(L, A) | 0.8
| L[:,None] * A | 1
| L @ A | 1
| norm(A) | 1.3
| A/1 | 1.4
| np.abs(A) | 3
| is_diag(A) | 3.5
| normalize(A) | 4
| np.all(A > 0) | 5

Level 2: Medium overhead, deactivate if used often
| function | time (ms) |
--|--
| np.allclose(A, A) | 9
| np.allclose(A, A.conj().T) | 19 (why??)
| is_hermitian | 19
| A @ A | 45
| is_unitary | 50

Level 3: Severe overhead, active only for interactive or critical calls
| function | time (ms) |
--|--
| inv | 200
| qr | 250
| eigvalsh | 550
| is_dm | 620
| eigh | 850
| svd | 1320
| pinv | 1370
| eigvals | 2700
| eig | 3750