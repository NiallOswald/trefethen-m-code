"""clencurt.py - Clenshaw-Curtis quadrature nodes and weights"""

import numpy as np


def clencurt(n):
    """Return Clenshaw-Curtis quadrature nodes and weights."""
    theta = np.pi * np.arange(n + 1) / n
    x = np.cos(theta)
    w = np.zeros(n + 1)
    ii = np.arange(1, n)
    v = np.ones(n - 1)

    if n % 2:
        w[0] = 1 / (n**2)
        w[n] = w[0]
        for k in range(1, (n + 1) // 2):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1)
    else:
        w[0] = 1 / (n**2 - 1)
        w[n] = w[0]
        for k in range(1, n // 2):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1)
        v -= np.cos(n * theta[ii]) / (n**2 - 1)

    w[ii] = 2 * v / n

    return x, w
