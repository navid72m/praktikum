import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
from matplotlib.gridspec import GridSpec
N = 1000
X = []
for i in range(N):
    X.append((math.cos(2*math.pi*i/(N+1)),math.sin(2*math.pi*i/(N+1))))



data = np.array(X)
D = np.zeros((N,N))
for i in range(N):
    for j in range(N):
       D[i,j] =np.linalg.norm(data[i,:]-data[j,:])

epsilon=np.amax(D)*0.05
W=-np.sqrt(D)
W=W/epsilon
W=np.exp(W)
sumW=np.sum(W, axis=1)

P= np.diag(sumW)
P_inverse= np.linalg.inv(P)
K=np.dot(np.dot(P_inverse,W),P_inverse )
sumK=np.sum(K, axis=1)
Q=np.diag(sumK)
Qpwr= fractional_matrix_power(Q,-0.5)
T_hat=np.dot(np.dot(Qpwr,K),Qpwr)
eigenValues, eigenVectors = np.linalg.eig(T_hat)
print(T_hat.shape)
plt.plot(data[:,0],data[:,1])
plt.show()

L=5
A, V = np.linalg.eig(T_hat)
Al = A[0:L+1]
Vl=V[:,:L+1]
Lambdal = np.sqrt(np.power(Al,1/epsilon))
#print(Lambdal)
Q_inverse = np.linalg.inv(Q)
#T = np.dot(Q_inverse,K)
#a,phi = np.linalg.eig(T)
#phi_l = phi[:,:L+1]
Qpwr= fractional_matrix_power(Q,-0.5)
phi_l=np.dot(Qpwr,Vl)

Tk = []
for k in range(N):
    Tk.append(2*math.pi*k/(N+1))
print(phi_l[:,0].shape)

fig = plt.figure(constrained_layout=True)
gs = GridSpec(3, 2, figure=fig)

ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(phi_l[:,0],Tk)
ax0.set_xlim=((0,10))
ax0.set_xlabel('x')
ax0.set_ylabel('\u03C6(x)')

ax1=fig.add_subplot(gs[0, 1])
ax1.plot(phi_l[:,1],Tk)
ax1.set_ylabel('\u03C61(x)')
ax1.set_xlabel('x')

ax2=fig.add_subplot(gs[1,0])
ax2.plot(phi_l[:,2],Tk)
ax2.set_ylabel('\u03C62(x)')
ax2.set_xlabel('x')

ax3=fig.add_subplot(gs[1,1])
ax3.plot(phi_l[:,3],Tk)
ax3.set_ylabel('\u03C63(x)')
ax3.set_xlabel('x')

ax4=fig.add_subplot(gs[2,0])
ax4.plot(phi_l[:,4],Tk)
ax4.set_ylabel('\u03C64(x)')
ax4.set_xlabel('x')

ax5=fig.add_subplot(gs[2,1])
ax5.plot(phi_l[:,5],Tk)
ax5.set_ylabel('\u03C65(x)')
ax5.set_xlabel('x')
plt.show()