### Check levels

Let `A` be a matrix of `size = 2**10`, `dtype = complex`, and appropriate type (e.g. hermitian).
Let `L` be a vector of `size = 2**10` and `dtype = complex`.

Level 0: Virtually instant, no restriction
| function | time (ms) |
--|--
| A.T | 0.00005
| A.ravel() | 0.00005
| A.real | 0.00005
| A.reshape([2]*20) | 0.0004
| L.copy() | 0.0005
| L.conj() | 0.00055
| np.diag(A) | 0.0006
| L**2 | 0.0007
| L @ L | 0.0008
| np.trace(A) | 0.0013
| L.conj() @ L | 0.0014
| np.sum(L) | 0.0016
| L/1 | 0.0019
| np.abs(L) | 0.002
| np.all(L > 0) | 0.0025
| is_ket(L) | 0.0028
| allclose0(L) | 0.003
| norm(L) | 0.0048
| normalize(L) | 0.0052
| L**4 | 0.0058
| np.allclose(L, L) | 0.015

Level 1: Minimal burden, disable only for high performance
| function | time (ms) |
--|--
| np.all(A.real > 0) | 0.45
| A**2 | 0.55
| A.copy() | 0.58
| A.conj() | 0.6
| allclose0(A, 0) | 0.7
| is_eigenstate(L, A) | 0.8
| L[:,None] * A | 1
| L @ A | 1
| allclose0(A) | 1.25
| trace_product(A, A) | 1.2
| norm(A) | 1.3
| A/1 | 1.4
| is_diag(A) | 1.6
| np.abs(A) | 3.2
| A.T.copy() | 3.8
| normalize(A) | 3.6
| np.all(A > 0) | 4.9
| A**4 | 5.3

Level 2: Medium overhead, deactivate if used often
| function | time (ms) |
--|--
| is_hermitian(A) | 8
| np.allclose(A, A) | 9
| np.allclose(A, A.T.copy()) | 13.5
| np.allclose(A, A.T) | 17.5
| A @ A | 42
| is_unitary(A) | 50

Level 3: Severe overhead, active only for interactive or critical calls
(Functions are mostly from scipy.linalg)
| function | time (ms) |
--|--
| inv | 200
| qr | 250
| eigvalsh | 550
| is_dm | 560
| eigh | 850
| svd | 1300
| pinv | 1350
| polar | 1390
| eigvals | 2700
| schur | 3650
| eig | 3750