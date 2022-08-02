"""p18.py - Chebyshev differentiation via FFT (compare p11.py)"""

import numpy as np
from chebfft import chebfft
import matplotlib.pyplot as plt

xx = np.arange(-1, 1.005, 0.01)
ff = np.exp(xx) * np.sin(5 * xx)

for n in [10, 20]:
    x = np.cos(np.pi * np.arange(0, n + 1) / n)
    f = np.exp(x) * np.sin(5 * x)
    plt.subplot(2, 2, n // 5 - 1)
    plt.plot(x, f, "k.", markersize=8)
    plt.plot(xx, ff, "k-", linewidth=0.75)
    plt.grid()
    plt.title(f"$f(x), N = {n}$")

    error = chebfft(f) - np.exp(x) * (np.sin(5 * x) + 5 * np.cos(5 * x))
    plt.subplot(2, 2, n // 5)
    plt.plot(x, error, "k.", markersize=8)
    plt.plot(x, error, "k-", linewidth=0.75)
    plt.grid()
    plt.title(f"error in $f'(x), N = {n}$")

plt.tight_layout()
plt.show()
