"""p13.py - solve linear BVP u_xx = exp(4x), u(-1)=u(1)=0"""

import numpy as np
from cheb import cheb
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 3.5))

n = 16
D, x = cheb(n)
D2 = D @ D
D2 = D2[1:n, 1:n]
f = np.exp(4 * x[1:n])
u = np.linalg.solve(D2, f)
u = np.concatenate([[0], u, [0]])

plt.plot(x, u, "k.", markersize=8)
xx = np.arange(-1, 1.01, 0.01)
uu = np.polyval(np.polyfit(x, u, n), xx)
plt.plot(xx, uu, "k-", linewidth=0.75)
plt.grid()
plt.xticks(np.arange(-1, 1.2, 0.2))
exact = (np.exp(4 * xx) - np.sinh(4) * xx - np.cosh(4)) / 16
plt.title(f"max err = {np.linalg.norm(uu - exact, np.inf):.3e}")

plt.show()
