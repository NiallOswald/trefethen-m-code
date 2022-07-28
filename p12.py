"""p12.py - accuracy of Chebyshev spectral differentiation (compare p7.py)"""

import numpy as np
from cheb import cheb
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

# Compute derivatives for various values of n
n_max = 50
E = np.zeros((4, n_max))

for n in range(1, n_max + 1):
    D, x = cheb(n)
    v = abs(x) ** 3
    dv = 3 * x * abs(x)
    E[0, n - 1] = np.linalg.norm(D @ v - dv, np.inf)
    v = np.exp(-(x ** (-2)))
    dv = 2 * v / x**3
    E[1, n - 1] = np.linalg.norm(D @ v - dv, np.inf)
    v = 1 / (1 + x**2)
    dv = -2 * x * v**2
    E[2, n - 1] = np.linalg.norm(D @ v - dv, np.inf)
    v = x**10
    dv = 10 * x**9
    E[3, n - 1] = np.linalg.norm(D @ v - dv, np.inf)

# Plot results
titles = ["$|x^3|$", "$\\exp(-x^{-2})$", "$1/(1+x^2)$", "$x^{10}$"]
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(np.arange(1, n_max + 1), E[i, :], "k.", markersize=6)
    plt.plot(np.arange(1, n_max + 1), E[i, :], "k-", linewidth=0.75)

    plt.yscale("log")
    plt.xlim(0, n_max)
    plt.ylim(1e-16, 1e3)
    plt.grid()
    plt.xticks(np.arange(0, n_max + 1, 10))
    plt.yticks(10.0 ** np.arange(-15, 5, 5))
    plt.xlabel("N")
    plt.ylabel("error")
    plt.title(titles[i])

plt.tight_layout()
plt.show()
