# p8 - eigenvalues of harmonic oscillator -u''+x^2 u on R

import numpy as np
import scipy.linalg as sp

L = 8
for n in range(6, 37, 6):
    h = 2 * np.pi / n
    x = np.arange(1, n + 1) * h
    x = L * (x - np.pi) / np.pi

    column = np.concatenate(
        [
            [-np.pi**2 / (3 * h**2) - 1 / 6],
            -0.5
            * (-1) ** np.arange(1, n)
            / np.sin(h * np.arange(1, n) / 2) ** 2,
        ]
    )
    D2 = (np.pi / L) ** 2 * sp.toeplitz(column)

    eigenvalues = np.sort(sp.eigvalsh(-D2 + np.diag(x**2)))
    print(n, eigenvalues[:4])
