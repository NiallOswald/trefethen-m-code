"""p21.py - eigenvalues of Mathieu operator -u_xx + 2qcos(2x)u"""

import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt

n = 42
h = 2 * np.pi / n
x = np.arange(1, n + 1) * h

D2 = sp.toeplitz(
    np.concatenate(
        [
            [-np.pi**2 / (3 * h**2) - 1 / 6],
            -0.5
            * (-1.0) ** np.arange(1, n)
            / np.sin(h * np.arange(1, n) / 2) ** 2,
        ]
    )
)
qq = np.arange(0, 15.1, 0.2)
data = np.zeros((len(qq), 11))
for i, q in enumerate(qq):
    e = np.sort(sp.eigvals(-D2 + 2 * q * np.diag(np.cos(2 * x)))).real
    data[i, :] = e[:11]

fig = plt.figure(figsize=(6, 6))
ax = plt.subplot(1, 2, 2)
ax.plot(qq, data[:, ::2], "k-", linewidth=0.8)
ax.plot(qq, data[:, 1::2], "k--", linewidth=0.8)
ax.set_xlabel("$q$", fontsize=10)
ax.set_ylabel("$\\lambda$", fontsize=10)
ax.set_xlim(0, 15)
ax.set_ylim(-24, 32)
ax.set_yticks(np.arange(-24, 33, 4))

plt.tight_layout()
plt.show()
