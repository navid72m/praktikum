import numpy as np
import matplotlib.pyplot as plt
import math

from mpl_toolkits.mplot3d import Axes3D


def lorenz(x, y, z, s=10, r=28, b=(8/3)):

    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def distance(px,py,pz,px0,py0,pz0):
    return math.sqrt(((px-px0)**2)+((py-py0)**2)+((pz-pz0)**2))
dt = 0.01
num_steps = 5000

# Need one more for the initial values
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

# Set initial values
xs[0], ys[0], zs[0] = (10., 10., 10.)

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
larger_than_one=False
for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)



# Need one more for the initial values
xs1 = np.empty(num_steps + 1)
ys1 = np.empty(num_steps + 1)
zs1 = np.empty(num_steps + 1)

# Set initial values
xs1[0], ys1[0], zs1[0] = (10+(10**-8), 10., 10.)

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point

for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs1[i], ys1[i], zs1[i])
    xs1[i + 1] = xs1[i] + (x_dot * dt)
    ys1[i + 1] = ys1[i] + (y_dot * dt)
    zs1[i + 1] = zs1[i] + (z_dot * dt)

for i in range(num_steps):
    if larger_than_one==False and distance(xs[i],ys[i],zs[i],xs1[i],ys1[i],zs1[i])>1:
        print(i)
        larger_than_one = True


# Plot
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(xs, ys, zs, lw=0.5)
ax.plot(xs1, ys1, zs1, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()
