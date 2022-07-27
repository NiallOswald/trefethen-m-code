# p11 - Chebyshev differentiation of a smooth function

import numpy as np
from cheb import cheb
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

xx = np.arange(-1, 1.01, 0.01)
uu = np.exp(xx) * np.sin(5 * xx)

for n in [10, 20]:
    D, x = cheb(n)
    u = np.exp(x) * np.sin(5 * x)
    plt.subplot(2, 2, 2 * (n // 10) - 1)
    plt.plot(x, u, "k.", markersize=8)
    plt.grid()
    plt.plot(xx, uu, "k-", linewidth=0.75)
    plt.xlim(-1, 1)
    plt.ylim(-4, 2)
    plt.yticks(np.arange(-4, 3, 2))
    plt.title(f"$u(x)$, $N={n}$")

    error = D @ u - np.exp(x) * (np.sin(5 * x) + 5 * np.cos(5 * x))
    plt.subplot(2, 2, 2 * (n // 10))
    plt.plot(x, error, "k.", markersize=8)
    plt.grid()
    plt.plot(x, error, "k-", linewidth=0.75)
    plt.xlim(-1, 1)
    plt.title(f"error in $u''(x)$, $N={n}$")

plt.tight_layout()
plt.show()
