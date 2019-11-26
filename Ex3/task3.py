import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

w = 3



fig = plt.figure(figsize=(7, 9))
gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 1])

#  two real negative tools
alpha = -1.8
beta = 2
Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U =  X*alpha -Y-X*((X**2)+Y)**2
V =  X+alpha*Y-Y*((X**2)+Y)**2

ax0 = fig.add_subplot(gs[0, 0])
ax0.streamplot(X, Y, U, V, density=[0.9, 1])
ax0.set_title('\u03B1=-1.8' ,fontsize= 8)

alpha1 = 0
Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U =  X*alpha1 -Y-X*((X**2)+Y)**2
V =  X+alpha1*Y-Y*((X**2)+Y)**2

ax1 = fig.add_subplot(gs[0, 1])
ax1.streamplot(X, Y, U, V, density=[0.9, 1])
ax1.set_title('\u03B1=0' ,fontsize= 8)

alpha2 = 1.3
Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U =  X*alpha2 -Y-X*((X**2)+Y)**2
V =  X+alpha2*Y-Y*((X**2)+Y)**2

ax2 = fig.add_subplot(gs[1, 0])
ax2.streamplot(X, Y, U, V, density=[0.9, 1])
ax2.set_title('\u03B1=1.3' ,fontsize= 8)

# Varying color along a streamline


#  Varying line width along a streamline









plt.tight_layout()
plt.show()