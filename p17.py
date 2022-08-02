"""p17.py - Helmholtz eq. u_xx + u_yy + (k^2)u = f on [-1,1]x[-1,1]"""

import numpy as np
import scipy.interpolate as sp
from cheb import cheb
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D  # noqa: F401

# Set up spectral grid and tensor product Helmholtz operator
n = 24
D, x = cheb(n)

k = 9
y = x
xx, yy = np.meshgrid(x[1:n], y[1:n])
xx = xx.flatten()
yy = yy.flatten()
f = np.exp(-10 * ((yy - 1) ** 2 + (xx - 0.5) ** 2))

D2 = D @ D
D2 = D2[1:n, 1:n]
I = np.eye(n - 1)  # noqa: E741
L = np.kron(I, D2) + np.kron(D2, I) + k**2 * np.eye((n - 1) ** 2)

# Solve for u, reshape to 2D grid, and plot
u = np.linalg.solve(L, f)
uu = np.zeros((n + 1, n + 1))
uu[1:n, 1:n] = np.reshape(u, (n - 1, n - 1))

xx, yy = np.meshgrid(x, y)
x_fine = np.arange(-1, 1.01, 0.0333)
y_fine = x_fine
xxx, yyy = np.meshgrid(x_fine, y_fine)
uuu = sp.interp2d(xx, yy, uu, kind="cubic")(x_fine, y_fine)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(xxx, yyy, uuu, rstride=1, cstride=1, cmap="viridis")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$u$")
ax.text(0.2, 1, 0.022, f"$u(0, 0) = {uu[n//2, n//2]:.11f}$")
ax.set_zlim(-0.03, 0.03)

plt.show()

fig = plt.figure()
plt.contour(
    xxx, yyy, uuu, cmap="viridis", levels=np.linspace(-0.025, 0.025, 11)
)
plt.xlim(-1, 1)
plt.ylim(-1, 1)

plt.show()
