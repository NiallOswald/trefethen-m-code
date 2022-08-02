"""CHEB  compute D = differentiation matrix, x = Chebyshev grid"""

import numpy as np


def cheb(n):
    """Return the Chebyshev differentiation matrix of order n."""
    if n == 0:
        return 0, 1

    x = np.array([np.cos(np.pi * np.arange(n + 1) / n)]).T
    c = np.array(
        [np.concatenate([[2], np.ones(n - 1), [2]]) * (-1) ** np.arange(n + 1)]
    ).T
    X = np.tile(x, (1, n + 1))  # noqa: N806
    dX = X - X.T  # noqa: N806
    D = (c @ (1 / c).T) / (dX + np.eye(n + 1))  # noqa: N806
    D = D - np.diag(np.sum(D.T, axis=0))  # noqa: N806

    return D, x.T[0]
