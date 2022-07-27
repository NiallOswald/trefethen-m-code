"""p5.py - repetition of p4.py via FFT"""

# For complex v, delete "np.real" commands

import numpy as np
import matplotlib.pyplot as plt

# Setup plots
fig, axs = plt.subplots(2, 2, figsize=(8, 4))

# Differentiation of a hat function
n = 24
h = 2 * np.pi / n
x = h * np.arange(1, n + 1)

v = np.maximum(0, 1 - abs(x - np.pi) / 2)
v_hat = np.fft.fft(v)
w_hat = (
    1j
    * np.concatenate([np.arange(0, n / 2), [0], np.arange(1 - n / 2, 0)])
    * v_hat
)
w = np.real(np.fft.ifft(w_hat))

axs[0, 0].plot(x, v, "k.-", markersize=8, linewidth=0.75)
axs[0, 0].set_xlim(0, 2 * np.pi)
axs[0, 0].set_ylim(-0.5, 1.5)
axs[0, 0].grid(which="both")
axs[0, 0].set_title("function")

axs[0, 1].plot(x, w, "k.-", markersize=8, linewidth=0.75)
axs[0, 1].set_xlim(0, 2 * np.pi)
axs[0, 1].set_ylim(-1, 1)
axs[0, 1].grid(which="both")
axs[0, 1].set_title("spectral derivative")

# Differentiation of exp(sin(x))
v = np.exp(np.sin(x))
dv = np.cos(x) * v
v_hat = np.fft.fft(v)
w_hat = (
    1j
    * np.concatenate([np.arange(0, n / 2), [0], np.arange(1 - n / 2, 0)])
    * v_hat
)
w = np.real(np.fft.ifft(w_hat))

axs[1, 0].plot(x, v, "k.-", markersize=8, linewidth=0.75)
axs[1, 0].set_xlim(0, 2 * np.pi)
axs[1, 0].set_ylim(0, 3)
axs[1, 0].grid(which="both")

axs[1, 1].plot(x, w, "k.-", markersize=8, linewidth=0.75)
axs[1, 1].set_xlim(0, 2 * np.pi)
axs[1, 1].set_ylim(-2, 2)
axs[1, 1].grid(which="both")

error = np.linalg.norm(w - dv, np.inf)
axs[1, 1].text(4.2, 1.4, f"max error = {error:.4e}", ha="center", va="center")

plt.show()
