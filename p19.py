"""p19.py - 2nd-order wave eq. on Chebyshev grid (compare p6.m)"""

import numpy as np
from chebfft import chebfft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.collections import LineCollection
from alive_progress import alive_bar

# Time-stepping by leap frog formula
n = 80
x = np.cos(np.pi * np.arange(0, n + 1) / n)
dt = 8 / n**2
v = np.exp(-200 * x**2)
v_old = np.exp(-200 * (x - dt) ** 2)
t_max = 4
t_plot = 0.075
plot_gap = round(t_plot / dt)
dt = t_plot / plot_gap
n_plots = round(t_max / t_plot)
data = np.zeros((n_plots + 1, n + 1))
data[0, :] = v
t_data = np.zeros(1)

with alive_bar(n_plots) as bar:
    for i in range(n_plots):
        for j in range(plot_gap):
            w = chebfft(chebfft(v))
            w[0] = 0
            w[n] = 0
            v_new = 2 * v - v_old + dt**2 * w
            v_old = v
            v = v_new
        data[i + 1, :] = v
        t_data = np.append(t_data, dt * i * plot_gap)
        bar()

# Manipulate data for plotting as no waterfall method exists
template = np.tile(x, (2, 1)).T
plot_data = np.tile(template, (n_plots + 1, 1, 1))
plot_data[:, :, 1] = data[:, :]

# Setup figure
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(projection="3d")

# Plot results
line = LineCollection(plot_data, color="black", linewidth=0.5)
ax.add_collection3d(line, zs=t_data, zdir="y")
ax.set_zticks((-2, 0, 2))
ax.set_xlim(-1, 1)
ax.set_ylim(0, t_max)
ax.set_zlim(-2, 2)
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u")
ax.grid(False)

# Fix to stretch z axis
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.2, 1]))
plt.show()
