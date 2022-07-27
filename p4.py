"""p4.py - periodic spectral differentiation"""

import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt

# Setup plots
fig, axs = plt.subplots(2, 2, figsize=(8, 4))

# Set up grid and differentiation matrix
n = 24
h = 2 * np.pi / n
x = h * np.arange(1, n + 1)
column = np.concatenate(
    [[0], 0.5 * (-1) ** np.arange(1, n) / np.tan(np.arange(1, n) * h / 2)]
)
D = sp.toeplitz(column, column[np.concatenate([[0], np.arange(n - 1, 0, -1)])])

# Differentiation of a hat function
v = np.maximum(0, 1 - abs(x - np.pi) / 2)
axs[0, 0].plot(x, v, "k.-", markersize=8, linewidth=0.75)
axs[0, 0].set_xlim(0, 2 * np.pi)
axs[0, 0].set_ylim(-0.5, 1.5)
axs[0, 0].grid(which="both")
axs[0, 0].set_title("function")

axs[0, 1].plot(x, D @ v, "k.-", markersize=8, linewidth=0.75)
axs[0, 1].set_xlim(0, 2 * np.pi)
axs[0, 1].set_ylim(-1, 1)
axs[0, 1].grid(which="both")
axs[0, 1].set_title("spectral derivative")

# Differentiation of exp(sin(x))
v = np.exp(np.sin(x))
dv = np.cos(x) * v
axs[1, 0].plot(x, v, "k.-", markersize=8, linewidth=0.75)
axs[1, 0].set_xlim(0, 2 * np.pi)
axs[1, 0].set_ylim(0, 3)
axs[1, 0].grid(which="both")

axs[1, 1].plot(x, D @ v, "k.-", markersize=8, linewidth=0.75)
axs[1, 1].set_xlim(0, 2 * np.pi)
axs[1, 1].set_ylim(-2, 2)
axs[1, 1].grid(which="both")

error = np.linalg.norm(D @ v - dv, np.inf)
axs[1, 1].text(4.2, 1.4, f"max error = {error:.4e}", ha="center", va="center")

plt.show()
