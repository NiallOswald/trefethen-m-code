"""p15.py - solve eigenvalue BVP u_xx = lambda*u, u(-1)=u(1)=0"""

import numpy as np
from cheb import cheb
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 6))

n = 36
D, x = cheb(n)
D2 = D @ D
D2 = D2[1:n, 1:n]

# Sort eigenvalues and -vectors
eigs, V = np.linalg.eig(D2)
ii = np.argsort(-eigs)
eigs = eigs[ii]
V = -V[:, ii]

# Plot 6 eigenvectors
for j in range(4, 34, 5):
    u = np.concatenate([[0], V[:, j], [0]])
    plt.subplot(7, 1, (j + 1) // 5)
    plt.plot(x, u, "k.", markersize=8)
    plt.grid()
    xx = np.arange(-1, 1.01, 0.01)
    uu = np.polyval(np.polyfit(x, u, n), xx)
    plt.plot(xx, uu, "k-", linewidth=0.75)
    plt.axis("off")
    plt.text(-0.4, 0.5, f"eig {j + 1} $= {eigs[j]*4/np.pi**2:.13f}*\\pi^2/4$")
    plt.text(0.7, 0.5, f"{4*n/(np.pi * (j + 1)):.1f} ppw")

plt.tight_layout()
plt.show()
