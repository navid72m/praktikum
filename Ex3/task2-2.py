import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

def fun1(alpha):
    return math.sqrt((alpha/2)-1)
def fun2(alpha):
    return -math.sqrt((alpha/2 )-1)
n = 10000
r = np.linspace(2,4 , n)

iterations = 1000
last = 100

x = np.linspace(-1,1,n)

fig = plt.figure(figsize=(7, 9))
gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 1])
ax0 = fig.add_subplot(gs[:, :])

y=[]
y2=[]
for element in r:
    y.append(fun1(element))
for element in r:
    y2.append(fun2(element))
ax0.plot(r, y, ',k', alpha=.25)
ax0.plot(r, y2, ',k', alpha=.25)

ax0.set_xlim(-1,4)



ax0.set_title("Bifurcation diagram")

plt.tight_layout()
plt.show()

