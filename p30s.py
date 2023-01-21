"""p30.py - Spectral integration, Clenshaw-Curtis style"""

import numpy as np
from scipy.special import erf
from clencurt import clencurt
import matplotlib.pyplot as plt

# Computation: various values of N, four functions
n_max = 50
e = np.zeros((4, n_max))

for n in range(1, n_max + 1):
    x, w = clencurt(n)

    f = np.abs(x) ** 3
    e[0, n - 1] = abs(np.dot(w, f) - 0.5)

    f = np.exp(-(x ** (-2)))
    e[1, n - 1] = abs(
        np.dot(w, f) - 2 * (np.exp(-1) + np.sqrt(np.pi) * (erf(1) - 1))
    )

    f = 1 / (1 + x**2)
    e[2, n - 1] = abs(np.dot(w, f) - np.pi / 2)

    f = x**10
    e[3, n - 1] = abs(np.dot(w, f) - 2 / 11)

# Plot results
fig = plt.figure(figsize=(8, 5))
labels = ["$|x|^3$", "$e^{-x^{-2}}$", "$1/(1+x^2)$", "$x^{10}$"]

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.semilogy(e[i, :] + 1e-100, "k.")
    plt.plot(e[i, :] + 1e-100, "k-")
    plt.axis([0, n_max, 1e-18, 1e3])
    plt.grid()
    plt.xticks(np.arange(0, n_max + 1, 10))
    plt.yticks(10.0 ** np.arange(-15, 1, 5))
    plt.ylabel("error")
    plt.text(0.75, 0.75, labels[i], fontsize=10, transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()
