"""p20.py - 2nd-order wave eq. in 2D via FFT (compare p19.py)"""

import numpy as np
import scipy.interpolate as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from alive_progress import alive_bar

fig = plt.figure(figsize=(8, 8))

# Grid and initial data
n = 24
x = np.cos(np.pi * np.arange(0, n + 1) / n)
y = x
dt = 6 / n**2
xx, yy = np.meshgrid(x, y)
plot_gap = round((1 / 3) / dt)
dt = (1 / 3) / plot_gap
vv = np.exp(-40 * ((xx - 0.4) ** 2 + yy**2))
print(np.max(vv))
vv_old = vv

# Time-stepping by leap frog formula
with alive_bar(3 * plot_gap + 1) as bar:
    for k in range(3 * plot_gap + 1):
        t = k * dt

        # Generate plots at multiples of t = 1/3
        if (k + 0.5) % plot_gap < 1:
            i = k // plot_gap + 1
            ax = fig.add_subplot(2, 2, i, projection="3d")
            x_fine = np.arange(-1, 1.01, 1 / 16)
            y_fine = x_fine
            xxx, yyy = np.meshgrid(x_fine, y_fine)
            vvv = sp.griddata(
                (xx.ravel(), yy.ravel()),
                vv.ravel(),
                (xxx, yyy),
                method="cubic",
            )
            ax.plot_surface(
                xxx, yyy, vvv, cmap="viridis", rstride=1, cstride=1
            )
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-0.15, 1)
            ax.set_title(f"t = {t}", fontsize=10)
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
            ax.set_zticks([0, 0.5, 1])
            ax.set_xlabel("$x$", fontsize=10)
            ax.set_ylabel("$y$", fontsize=10)
            ax.set_zlabel("$u$", fontsize=10)

        uxx = np.zeros((n + 1, n + 1))
        uyy = np.zeros((n + 1, n + 1))
        ii = np.arange(1, n)

        for i in ii:
            v = vv[i, :]
            V = np.concatenate([v, v[ii][::-1]])
            U = np.fft.fft(V).real
            W1 = np.fft.ifft(
                1j
                * np.concatenate([np.arange(n), [0], np.arange(1 - n, 0)])
                * U
            ).real
            W2 = np.fft.ifft(
                (-1)
                * np.concatenate([np.arange(n + 1), np.arange(1 - n, 0)]) ** 2
                * U
            ).real
            uxx[i, ii] = W2[ii] / (1 - x[ii] ** 2) - x[ii] * W1[ii] / (
                1 - x[ii] ** 2
            ) ** (3 / 2)

        for j in ii:
            v = vv[:, j].T
            V = np.concatenate([v, v[ii][::-1]])
            U = np.fft.fft(V).real
            W1 = np.fft.ifft(
                1j
                * np.concatenate([np.arange(n), [0], np.arange(1 - n, 0)])
                * U
            ).real
            W2 = np.fft.ifft(
                (-1)
                * np.concatenate([np.arange(n + 1), np.arange(1 - n, 0)]) ** 2
                * U
            ).real
            uyy[ii, j] = W2[ii] / (1 - y[ii] ** 2) - y[ii] * W1[ii] / (
                1 - y[ii] ** 2
            ) ** (3 / 2)

        vv_new = 2 * vv - vv_old + dt**2 * (uxx + uyy)
        vv_old = vv
        vv = vv_new

        bar()

plt.tight_layout()
plt.show()
