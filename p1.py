import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

n_vals = 2 ** np.arange(3, 13)
times = np.zeros(len(n_vals))

for i, n in enumerate(n_vals):
    h = 2 * np.pi / n
    x = np.arange(1, n + 1) * h - np.pi
    u = np.exp(np.sin(x))
    du = np.cos(x) * u

    # Construct the matrix
    e = np.ones(n)
    D = sp.csr_matrix(
        (2 * e / 3, (np.arange(n), np.concatenate((np.arange(1, n), [0])))),
        shape=(n, n),
    )
    D = D - sp.csr_matrix(
        (e / 12, (np.arange(n), np.concatenate((np.arange(2, n), [0], [1])))),
        shape=(n, n),
    )
    D = (D - D.T) / h

    # Compute errors
    error = np.linalg.norm(D @ u - du, np.inf)

    plt.loglog(n, error, ".", markersize=8)

plt.xlabel("N")
plt.ylabel("error")
plt.title("Convergence of fourth-order finite differences")
plt.grid(which="both")
plt.plot(n_vals, n_vals ** (-4.0), "c--")
plt.show()
