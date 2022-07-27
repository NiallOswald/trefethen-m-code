"""p3.py - Band-limited interpolation"""

import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 1, sharex=True)

h = 1
x_max = 10
x = np.arange(-x_max, x_max + h, h)
xx = np.arange(-x_max - h / 20, x_max + 3 * h / 20, h / 10)

for i in range(3):
    if i == 0:
        v = x == 0
    elif i == 1:
        v = abs(x) <= 3
    elif i == 2:
        v = np.maximum(np.zeros(len(x)), 1 - abs(x) / 3)

    axs[i].plot(x, v, "k.", markersize=8)

    p = np.zeros(len(xx))
    for j in range(len(x)):
        p += v[j] * np.sin(np.pi * (xx - x[j]) / h) / (np.pi * (xx - x[j]) / h)

    axs[i].plot(xx, p, "k", linewidth=0.75)
    axs[i].set_xlim(-x_max, x_max)
    axs[i].set_ylim(-0.5, 1.5)
    axs[i].set_xticks([])
    axs[i].set_yticks([0, 1])
    axs[i].grid(which="both")

plt.show()
