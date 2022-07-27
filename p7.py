"""p7.py - accuaracy of periodic spectral differentiation"""

import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt

# Compute derivatives for various values of N:
n_max = 50
E = np.zeros((4, n_max // 2 - 2))
for n in range(6, n_max + 1, 2):
    h = 2 * np.pi / n
    x = np.arange(1, n + 1) * h
    column = np.concatenate(
        [[0], 0.5 * (-1) ** np.arange(1, n) / np.tan(np.arange(1, n) * h / 2)]
    )
    D = sp.toeplitz(
        column, column[np.concatenate([[0], np.arange(n - 1, 0, -1)])]
    )

    v = abs(np.sin(x)) ** 3
    dv = 3 * np.sin(x) * np.cos(x) * abs(np.sin(x))
    E[0, n // 2 - 3] = np.linalg.norm(D @ v - dv, np.inf)

    v = np.exp(-np.sin(x / 2) ** (-2))
    dv = 0.5 * v * np.sin(x) / np.sin(x / 2) ** 4
    E[1, n // 2 - 3] = np.linalg.norm(D @ v - dv, np.inf)

    v = 1 / (1 + np.sin(x / 2) ** 2)
    dv = -np.sin(x / 2) * np.cos(x / 2) * v**2
    E[2, n // 2 - 3] = np.linalg.norm(D @ v - dv, np.inf)

    v = np.sin(10 * x)
    dv = 10 * np.cos(10 * x)
    E[3, n // 2 - 3] = np.linalg.norm(D @ v - dv, np.inf)

# Plot results
titles = [
    "$|\\sin(x)|^3$",
    "$\\exp(-\\sin^{-2}(x/2))$",
    "$1/(1+\\sin^2(x/2))$",
    "$\\sin(10x)$",
]
plt.figure(figsize=(8, 6))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(
        np.arange(6, n_max + 1, 2),
        E[i, :],
        "k.-",
        markersize=8,
        linewidth=0.75,
    )
    plt.yscale("log")
    plt.xlim(0, n_max)
    plt.ylim(1e-16, 1e3)

    plt.grid(which="both")
    plt.xticks(np.arange(0, n_max + 1, 10))
    plt.yticks(10.0 ** np.arange(-15, 1, 5))
    plt.xlabel("$N$")
    plt.ylabel("error")
    plt.title(titles[i])

plt.tight_layout()
plt.show()
