"""p9.py - polynomial interpolation in equispaced and Chebyshev points"""

import numpy as np
import matplotlib.pyplot as plt

n = 16
xx = np.arange(-1.01, 1.015, 0.005)

plt.figure(figsize=(7, 3))

for i in range(2):
    if not i:
        s = "equispaced points"
        x = -1 + 2 * np.arange(n + 1) / n
    else:
        s = "Chebyshev points"
        x = np.cos(np.pi * np.arange(n + 1) / n)

    plt.subplot(1, 2, i + 1)

    u = 1 / (1 + 16 * x**2)
    uu = 1 / (1 + 16 * xx**2)
    p = np.polyfit(x, u, n)
    pp = np.polyval(p, xx)

    plt.plot(x, u, "k.", markersize=8)
    plt.plot(xx, pp, "k-", linewidth=0.75)

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1, 1.5)
    plt.title(s)

    error = np.linalg.norm(uu - pp, np.inf)
    plt.text(-0.5, -0.5, f"max error = {error:.5g}")

plt.tight_layout()
plt.show()
