"""p14.py - solve nonlinear BVP u_xx = exp(u), u(-1)=u(1)=0 (compare p14.py)"""

import numpy as np
from cheb import cheb
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))

n = 16
D, x = cheb(n)
D2 = D @ D
D2 = D2[1:n, 1:n]

u = np.zeros(n - 1)
change = 1
i = 0
while change > 1e-15:
    u_new = np.linalg.solve(D2, np.exp(u))
    change = np.linalg.norm(u_new - u, np.inf)
    u = u_new
    i += 1

u = np.concatenate([[0], u, [0]])
plt.plot(x, u, "k.", markersize=8)
xx = np.arange(-1, 1.01, 0.01)
uu = np.polyval(np.polyfit(x, u, n), xx)
plt.plot(xx, uu, "k-", linewidth=0.75)
plt.grid()
plt.xlim(-1, 1)
plt.ylim(-0.4, 0)
plt.xticks(np.arange(-1, 1.2, 0.2))
plt.yticks(np.arange(-0.4, 0.15, 0.1))  # wtaf???
plt.title(f"no. steps = {i}    $u(0) ={u[n//2]:.14f}$")
plt.show()
