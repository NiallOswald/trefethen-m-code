"""CHEBFFT  Chebyshev differentiation via FFT. Simple, not optimal."""

import numpy as np


def chebfft(v):
    """Return the derivative of v at the Chebyshev points using the FFT."""
    n = len(v) - 1
    if n == 0:
        return 0

    x = np.cos(np.arange(0, n + 1) * np.pi / n)
    ii = np.arange(0, n)
    V = np.concatenate([v, v[n - 1 : 0 : -1]])  # noqa: N806

    U = np.fft.fft(V).real  # noqa: N806
    W = np.fft.ifft(  # noqa: N806
        1j * np.concatenate([ii, [0], np.arange(1 - n, 0)]) * U
    ).real

    w = np.zeros(n + 1)
    w[1:n] = -W[1:n] / np.sqrt(1 - x[1:n] ** 2)
    w[0] = np.sum(ii**2 * U[ii]) / n + 0.5 * n * U[n]
    w[n] = (
        np.sum((-1) ** (ii + 1) * ii**2 * U[ii]) / n
        + 0.5 * (-1) ** (n + 1) * n * U[n]
    )

    return w
