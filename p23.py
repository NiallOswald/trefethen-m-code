"""p23.py - eigenvalues of perturbed Laplacian on [-1, 1]x[-1, 1]"""

import numpy as np
import scipy.interpolate as interp
from cheb import cheb
import matplotlib.pyplot as plt

# Set up tensor product Laplacian and compute 4 eigenmodes
n = 16
D, x = cheb(n)
y = x
xx, yy = np.meshgrid(x[1:n], y[1:n])
D2 = (D @ D)[1:n, 1:n]
I = np.eye(n - 1)  # noqa: E741
L = -np.kron(I, D2) - np.kron(D2, I)
L += np.diag(np.exp(20 * (yy - xx - 1)).ravel())
eigs, V = np.linalg.eig(L)
ii = np.argsort(eigs)
eigs = eigs[ii].real
ii = ii[:4]
V = V[:, ii].real

# Reshape them to 2D grid, interpolate to a finer grid, and plot
xx, yy = np.meshgrid(x, y)
fine = np.arange(-1.0, 1.01, 0.02)
xxx, yyy = np.meshgrid(fine, fine)
uu = np.zeros((n + 1, n + 1))
fig = plt.figure(figsize=(6, 6))

for i in range(4):
    uu[1:n, 1:n] = V[:, i].reshape(n - 1, n - 1)
    uu = uu / np.linalg.norm(uu.ravel(), np.inf)
    uuu = interp.griddata(
        (xx.ravel(), yy.ravel()), uu.ravel(), (xxx, yyy), method="cubic"
    )
    plt.subplot(2, 2, i + 1)
    plt.contour(
        xxx, yyy, uuu, levels=np.arange(-0.9, 1, 0.2), colors=["black"]
    )
    plt.axis("equal")
    plt.title(f"eig$= {eigs[i]/(np.pi**2/4):.12f}\\pi^2/4$", fontsize=10)

plt.tight_layout()
plt.show()
