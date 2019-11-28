import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

w = 3



fig = plt.figure(figsize=(7, 9))
gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 1])

#  two real negative tools
alpha = 0

Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U =  alpha - X**2
V =  -Y

ax0 = fig.add_subplot(gs[0, 0])
ax0.streamplot(X, Y, U, V, density=[0.5, 1])
ax0.set_title('\u03B1=0 ' ,fontsize= 8)

alpha1 = 1

Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U1 =  alpha1 - X**2
V1 =  -Y

ax1 = fig.add_subplot(gs[1, 0])
ax1.streamplot(X, Y, U1, V1, density=[0.5, 1])
ax1.set_title('\u03B1=1 ' ,fontsize= 8)

alpha2 = -1

Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U2 =  alpha2 - X**2
V2 =  -Y

ax2 = fig.add_subplot(gs[0, 1])
ax2.streamplot(X, Y, U2, V2, density=[0.5, 1])
ax2.set_title('\u03B1=-1 ' ,fontsize= 8)



alpha4 = 2

Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U4 =  alpha4 - X**2
V4 =  -Y

ax4 = fig.add_subplot(gs[1, 1])
ax4.streamplot(X, Y, U4, V4, density=[0.7, 1])
ax4.set_title('\u03B1=2 ' ,fontsize= 8)


w = 3



fig2 = plt.figure(figsize=(7, 9))
gs2 = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 1])

#  two real negative tools
alpha = 2

Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U =  alpha - 2*(X**2)-2
V =  -Y

ax0 = fig2.add_subplot(gs2[0, 1])
ax0.streamplot(X, Y, U, V, density=[0.5, 1])
ax0.set_title('\u03B1=2 ' ,fontsize= 8)

alpha1 = 1

Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U1 =  alpha1 - 2*(X**2)-2
V1 =  -Y

ax1 = fig2.add_subplot(gs2[1, 0])
ax1.streamplot(X, Y, U1, V1, density=[0.5, 1])
ax1.set_title('\u03B1=1 ' ,fontsize= 8)

alpha2 = -1

Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U2 =  alpha2 - 2*(X**2)-2
V2 =  -Y

ax2 = fig2.add_subplot(gs2[0, 0])
ax2.streamplot(X, Y, U2, V2, density=[0.5, 1])
ax2.set_title('\u03B1=-1 ' ,fontsize= 8)

alpha3 = 3

Y, X = np.mgrid[-w:w:100j, -w:w:100j]
U3 =  alpha3 - 2*(X**2)-2
V3 =  -Y

ax3 = fig2.add_subplot(gs2[1, 1])
ax3.streamplot(X, Y, U3, V3, density=[0.7, 1])
ax3.set_title('\u03B1=3 ' ,fontsize= 8)




plt.tight_layout()
plt.show()