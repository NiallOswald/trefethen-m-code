# p10 - polynomials and corresponding equipotential curves

import numpy as np
import matplotlib.pyplot as plt

n = 16

for i in range(2):
    if not i:
        s = "equispaced points"
        x = -1 + 2 * np.arange(n + 1) / n
    else:
        s = "Chebyshev points"
        x = np.cos(np.pi * np.arange(n + 1) / n)
    p = np.poly(x)

    # Plot p(x) over [-1, 1]
    xx = np.arange(-1, 1.005, 0.005)
    pp = np.polyval(p, xx)
    plt.subplot(2, 2, 2 * i + 1)
    plt.plot(x, 0 * x, "k.", markersize=6)
    plt.plot(xx, pp, "k-", linewidth=0.75)
    plt.grid(True)
    plt.xticks(np.arange(-1, 1.5, 0.5))
    plt.title(s)

    # Plot equipotential curves
    plt.subplot(2, 2, 2 * (i + 1))
    plt.plot(np.real(x), np.imag(x), "k.", markersize=6)
    plt.xlim(-1.4, 1.4)
    plt.ylim(-1.12, 1.12)

    xgrid = np.arange(-1.4, 1.42, 0.02)
    ygrid = np.arange(-1.12, 1.14, 0.02)
    xx, yy = np.meshgrid(xgrid, ygrid)
    zz = xx + 1j * yy
    pp = np.polyval(p, zz)
    levels = 10.0 ** np.arange(-4, 1)
    plt.contour(xx, yy, abs(pp), levels, colors="k", linewidths=0.5)
    plt.xticks(np.arange(-1, 1.5, 0.5))
    plt.title(s)

plt.tight_layout()
plt.show()
