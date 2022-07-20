"""p2.py - convergence of periodic spectral method (compare p1.py)"""

import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt

for n in range(2, 101, 2):
    h = 2 * np.pi / n
    x = np.arange(1, n + 1) * h - np.pi
    u = np.exp(np.sin(x))
    du = np.cos(x) * u

    # Construct the spectral differentiation matrix
    column = np.concatenate(
        [[0], 0.5 * (-1) ** np.arange(1, n) / np.tan(np.arange(1, n) * h / 2)]
    )
    index = np.concatenate([[0], np.arange(n - 1, 0, -1)])
    D = sp.toeplitz(column, column[index])

    # Compute errors
    error = np.linalg.norm(D @ u - du, np.inf)

    plt.loglog(n, error, "k.", markersize=8)

plt.xlabel("N")
plt.ylabel("error")
plt.title("Convergence of spectral differentiation")
plt.grid(which="both")
plt.show()
