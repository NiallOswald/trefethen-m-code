"""p6 - variable coefficient wave equation"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.collections import LineCollection

# Setup plot
fig = plt.figure()
ax = fig.gca(projection="3d")

# Grid, variable coefficient, and initial data
n = 128
h = 2 * np.pi / n
x = np.arange(1, n + 1) * h
t = 0
dt = h / 4
c = 0.2 + np.sin(x - 1) ** 2
v = np.exp(-100 * (x - 1) ** 2)
v_old = np.exp(-100 * (x - 0.2 * dt - 1) ** 2)

# Time-stepping by leap frog formula
t_max = 8
t_plot = 0.15
t_data = [t]
plot_gap = round(t_plot / dt)
dt = t_plot / plot_gap
n_plots = round(t_max / t_plot)
data = np.row_stack((v, np.zeros((n_plots, n))))
for i in range(n_plots):
    for j in range(plot_gap):
        t += dt
        v_hat = np.fft.fft(v)
        w_hat = (
            1j
            * np.concatenate(
                [np.arange(0, n / 2), [0], np.arange(1 - n / 2, 0)]
            )
            * v_hat
        )
        w = np.real(np.fft.ifft(w_hat))
        v_new = v_old - 2 * dt * c * w
        v_old = v
        v = v_new
    data[i + 1, :] = v
    t_data.append(t)

# Manipulate data for plotting as no waterfall method exists
template = np.tile(x, (2, 1)).T
data_plot = np.tile(template, (n_plots + 1, 1, 1))
data_plot[:, :, 1] = data[:, :]

# Make plot
line = LineCollection(data_plot, color="black")
ax.add_collection3d(line, zs=t_data, zdir="y")
ax.set_zticks((0, 2, 4))
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(0, t_max)
ax.set_zlim(0, 5)
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u")
ax.grid(False)

# Fix to stretch z axis
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.2, 1]))
plt.show()
