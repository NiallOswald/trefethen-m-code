"""p16.py - Poisson eq. on [-1, 1]x[-1, 1] with u=0 on the boundary"""

import numpy as np
import scipy.interpolate as sp
from cheb import cheb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from time import time

fig = plt.figure(figsize=(6, 6))

# Set up grids and tensor product Laplacian and solve for u
n = 24
D, x = cheb(n)
D2 = D @ D
D2 = D2[1:n, 1:n]

y = x
xx, yy = np.meshgrid(x[1:n], y[1:n])

# Flatten 2D grids to 1D vectors
xf = xx.flatten()
yf = yy.flatten()

f = 10 * np.sin(8 * xf * (yf - 1))
I = np.eye(n - 1)  # noqa: E741
L = np.kron(I, D2) + np.kron(D2, I)

plt.spy(L)
plt.show()

start = time()
u = np.linalg.solve(L, f)
end = time()
print("Time to solve:", end - start)

# Reshape u onto a 2D grid
uu = np.zeros((n + 1, n + 1))
uu[1:n, 1:n] = np.reshape(u, (n - 1, n - 1))
value = uu[n // 4, n // 4]

# Interpolate to a finer grid and plot
xx, yy = np.meshgrid(x, y)
x_fine = np.arange(-1, 1.01, 0.06)
y_fine = x_fine
xxx, yyy = np.meshgrid(x_fine, y_fine)
uuu = sp.griddata(
    (xx.ravel(), yy.ravel()), uu.ravel(), (xxx, yyy), method="cubic"
)
# uuu = sp.interp2d(xx, yy, uu, kind="cubic")(x_fine, y_fine)

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(xxx, yyy, uuu, rstride=1, cstride=1, cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u")
ax.text(0.4, -0.3, -0.3, "$u(2^{-1/2}, 2^{-1/2}) = " + f"{value:.11f}$")

plt.show()
