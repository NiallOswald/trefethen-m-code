"""p22.py - 5th eigenvector of Airy equation u_xx = lambda*x*u"""

import numpy as np
import scipy.linalg as la
import scipy.special as sp
from cheb import cheb
import matplotlib.pyplot as plt

for n in range(12, 49, 12):
    D, x = cheb(n)
    D2 = (D @ D)[1:n, 1:n]
    eigs, V = la.eig(D2, np.diag(x[1:n]))
    eigs = eigs.real
    ii = eigs > 0
    V = V[:, ii]
    eigs = eigs[ii]
    ii = np.argsort(eigs)
    ii = ii[4]
    eig = eigs[ii]
    v = np.concatenate([[0], V[:, ii], [0]])
    v = v / v[n // 2] * sp.airy(0)[0]
    xx = np.arange(-1, 1.01, 0.01)
    vv = np.polyval(np.polyfit(x, v, n), xx)
    plt.subplot(2, 2, n // 12)
    plt.plot(xx, vv, "k-", linewidth=0.75)
    plt.grid()
    plt.title(f"N = {n}    eig = {eig:.10f}", fontsize=10)

plt.tight_layout()
plt.show()
