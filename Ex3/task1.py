import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

w = 3



fig = plt.figure(figsize=(7, 9))
gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 1])

#  two real negative tools
alpha = -2
beta = 2
Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U =  X*alpha + Y*beta
V =  -0.25*X

ax0 = fig.add_subplot(gs[0, 0])
ax0.streamplot(X, Y, U, V, density=[0.5, 1])
ax0.set_title('\u03B1=-2 \u03B2=2 \u03BB1=-0.3 \u03BB1=-1.7' ,fontsize= 8)

alpha1 = -1
beta1 = 2
Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U1 =  X*alpha1 + Y*beta1
V1 =  -0.25*X

ax1 = fig.add_subplot(gs[1, 0])
ax1.streamplot(X, Y, U1, V1, density=[0.5, 1])
ax1.set_title('\u03B1=-1 \u03B2=2 \u03BB1=-0.5+0.5i \u03BB1=-0.5-0.5i' ,fontsize= 8)

alpha2 = 2
beta2 = -2
Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U2 =  X*alpha2 + Y*beta2
V2 =  -0.25*X

ax2 = fig.add_subplot(gs[0, 1])
ax2.streamplot(X, Y, U2, V2, density=[0.5, 1])
ax2.set_title('\u03B1=2 \u03B2=-2 \u03BB1=2.22 \u03BB1=-0.22' ,fontsize= 8)

alpha3 = 2
beta3 = 2
Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U3 =  X*alpha3 + Y*beta3
V3 =  -0.25*X

ax3 = fig.add_subplot(gs[1, 1])
ax3.streamplot(X, Y, U3, V3, density=[0.5, 1])
ax3.set_title('\u03B1=2 \u03B2=2 \u03BB1=1.7 \u03BB1=0.3' ,fontsize= 8)

alpha4 = 1
beta4 = 2
Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U4 =  X*alpha4 + Y*beta4
V4 =  -0.25*X

ax4 = fig.add_subplot(gs[2,0])
ax4.streamplot(X, Y, U4, V4, density=[0.5, 1])
ax4.set_title('\u03B1=1 \u03B2=2 \u03BB1=0.5+0.5i \u03BB1=0.5-0.5i' ,fontsize= 8)

# Varying color along a streamline


#  Varying line width along a streamline









plt.tight_layout()
plt.show()