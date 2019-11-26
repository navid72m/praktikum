import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def logistic(r, x):
    return r * x * (1 - x)

n = 10000
r = np.linspace(0, 4.0, n)

iterations = 1000
last = 100

x = 1e-5 * np.ones(n)

fig = plt.figure(figsize=(7, 9))
gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 1])
ax0 = fig.add_subplot(gs[:, :])

for i in range(iterations):
    x = logistic(r, x)

    if i >= (iterations - last):
        ax0.plot(r, x, ',k', alpha=.25)

x0 = -1e-5 * np.ones(n)

ax0.set_xlim(0, 4)
ax0.set_ylim(0,1)
ax0.set_title("Bifurcation diagram")

plt.tight_layout()
plt.show()

