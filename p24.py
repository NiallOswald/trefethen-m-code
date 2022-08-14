"""p24.py - psuedospectra of Davies' complex harmonic oscillator"""

import numpy as np
import scipy.linalg as la
from cheb import cheb
import matplotlib.pyplot as plt
from alive_progress import alive_bar

# Eigenvalues
n = 70
D, x = cheb(n)
x = x[1:n]

L = 6
x = L * x  # recale to [-L, L]
D = D / L

A = -D @ D
A = A[1:n, 1:n] + (1 + 3j) * np.diag(x**2)
eigs = np.linalg.eigvals(A)
plt.plot(eigs.real, eigs.imag, "k.", markersize=8)
plt.xlim(0, 50)
plt.ylim(0, 40)

# Pseudospectra
x = np.arange(0, 50.1, 2)  # For finer, slower plot, change 2 to 0.5
y = np.arange(0, 40.1, 2)
xx, yy = np.meshgrid(x, y)
zz = xx + 1j * yy
I = np.eye(n - 1)  # noqa: E741
sig_min = np.zeros((len(y), len(x)))

with alive_bar(len(x)) as bar:
    for j in range(len(x)):
        bar()
        for i in range(len(y)):
            sig_min[i, j] = la.svdvals(zz[i, j] * I - A)[-1]

plt.contour(
    x,
    y,
    sig_min,
    10.0 ** np.arange(-4, -0.25, 0.5),
    colors="black",
    linewidths=0.75,
)
plt.show()
