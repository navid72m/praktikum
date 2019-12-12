import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
from matplotlib.gridspec import GridSpec


# function for calculating the eigenvectors, and eigenvalues in diffusion map
def diffusion_map(L, N, data):
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = np.linalg.norm(data[i, :] - data[j, :])

    epsilon = np.amax(D) * 0.05
    W = np.power(D, 2)
    W = -1 * W
    W = W / epsilon
    W = np.exp(W)
    sumW = np.sum(W, axis=1)
    P = np.diag(sumW)
    P_inverse = np.linalg.inv(P)
    K = np.dot(np.dot(P_inverse, W), P_inverse)
    sumK = np.sum(K, axis=1)
    Q = np.diag(sumK)
    Qpwr = fractional_matrix_power(Q, -0.5)
    T_hat = np.dot(np.dot(Qpwr, K), Qpwr)
    A, V = np.linalg.eig(T_hat)
    Al = A[0:L + 1]
    Vl = V[:, :L + 1]
    lambda_l = np.sqrt(np.power(Al, 1 / epsilon))
    phi_l = np.dot(Qpwr, Vl)
    return phi_l, lambda_l


N = 1000
L = 5
X = []
for i in range(N):
    X.append((math.cos(2 * math.pi * i / (N + 1)), math.sin(2 * math.pi * i / (N + 1))))
data = np.array(X)
phi_l, lambda_l = diffusion_map(L, N, data)

# calculating the values for each tk
Tk = []
for k in range(N):
    Tk.append(2 * math.pi * k / (N + 1))

# plotting the phi_l(x) against tk

fig = plt.figure(constrained_layout=True)
gs = GridSpec(3, 2, figure=fig)

ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(Tk, phi_l[:, 0])
ax0.set_xlabel('x')
ax0.set_ylabel('\u03C60(x)')

ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(Tk, phi_l[:, 1])
ax1.set_ylabel('\u03C61(x)')
ax1.set_xlabel('x')

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(Tk, phi_l[:, 2])
ax2.set_ylabel('\u03C62(x)')
ax2.set_xlabel('x')

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(Tk, phi_l[:, 3])
ax3.set_ylabel('\u03C63(x)')
ax3.set_xlabel('x')

ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(Tk, phi_l[:, 4])
ax4.set_ylabel('\u03C64(x)')
ax4.set_xlabel('x')

ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(Tk, phi_l[:, 5])
ax5.set_ylabel('\u03C65(x)')
ax5.set_xlabel('x')
plt.savefig('xx.png')
plt.show()

from matplotlib.pyplot import figure
from sklearn import datasets

sr = datasets.make_swiss_roll(n_samples=1500, noise=0.0, random_state=None)
Y = np.array(sr[0])
phi_l, lambda_l = diffusion_map(10, 1500, Y)

# plotting the phi_l against phi_1 which is the first non-constant eigenvector
fig = plt.figure(figsize=(30, 20), constrained_layout=True)
gs = GridSpec(3, 3, figure=fig)
# figure(num=None, figsize=(8, 6), dpi=80)

ax0 = fig.add_subplot(gs[0, 0])
ax0.scatter(phi_l[:, 1], phi_l[:, 0])
ax0.set_xlabel('\u03C60(x)')
ax0.set_ylabel('\u03C61(x)')

ax1 = fig.add_subplot(gs[0, 1])
ax1.scatter(phi_l[:, 1], phi_l[:, 2])
ax1.set_ylabel('\u03C62(x)')
ax1.set_xlabel('\u03C61(x)')

ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(phi_l[:, 1], phi_l[:, 3])
ax2.set_ylabel('\u03C63(x)')
ax2.set_xlabel('\u03C61(x)')

ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(phi_l[:, 1], phi_l[:, 4])
ax3.set_ylabel('\u03C64(x)')
ax3.set_xlabel('\u03C61(x)')

ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(phi_l[:, 1], phi_l[:, 5])
ax4.set_ylabel('\u03C65(x)')
ax4.set_xlabel('\u03C61(x)')

ax5 = fig.add_subplot(gs[1, 2])
ax5.scatter(phi_l[:, 1], phi_l[:, 6])
ax5.set_ylabel('\u03C66(x)')
ax5.set_xlabel('\u03C61(x)')

ax6 = fig.add_subplot(gs[2, 0])
ax6.scatter(phi_l[:, 1], phi_l[:, 7])
ax6.set_ylabel('\u03C67(x)')
ax6.set_xlabel('\u03C61(x)')

ax7 = fig.add_subplot(gs[2, 1])
ax7.scatter(phi_l[:, 1], phi_l[:, 8])
ax7.set_ylabel('\u03C68(x)')
ax7.set_xlabel('\u03C61(x)')

ax8 = fig.add_subplot(gs[2, 2])
ax8.scatter(phi_l[:, 1], phi_l[:, 9])
ax8.set_ylabel('\u03C69(x)')
ax8.set_xlabel('\u03C61(x)')
plt.savefig("dif-2.png")
plt.show()
Y -= Y.mean(axis=0)
U, S, Vh = np.linalg.svd(Y, full_matrices=False, compute_uv=True)
# getting the direction of PCs
pc = Vh
# adding the magnitude to PCs by mutliplying by coresponding simgas
pc[0, :] *= S[0]
pc[1, :] *= S[1]
pc[2, :] *= S[2]
trace = np.sum(S)
e1 = S[0] / trace
e2 = S[1] / trace
e3 = S[2] / trace
print("The first PC: " + str(pc[0, :]))
print("Energy of first PC: " + str(e1))
print("The second  PC: " + str(pc[1, :]))
print("Energy of second PC: " + str(e2))
print("the third principle component: " + str(pc[2, :]))
print("Energy of third PC: " + str(e3))

import pandas as pd

df = pd.read_csv("data_DMAP_PCA_vadere.txt", header=None, delimiter=' ')

data = df.values

phi_l, lambda_l = diffusion_map(100, 1000, data)

fig = plt.figure(figsize=(30, 20), constrained_layout=True)
gs = GridSpec(3, 3, figure=fig)
# figure(num=None, figsize=(8, 6), dpi=80)

ax0 = fig.add_subplot(gs[0, 0])
ax0.scatter(phi_l[:, 1], phi_l[:, 0])
ax0.set_xlabel('\u03C61(x)')
ax0.set_ylabel('\u03C60(x)')

ax1 = fig.add_subplot(gs[0, 1])
ax1.scatter(phi_l[:, 1], phi_l[:, 2])
ax1.set_ylabel('\u03C62(x)')
ax1.set_xlabel('\u03C61(x)')

ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(phi_l[:, 1], phi_l[:, 3])
ax2.set_ylabel('\u03C63(x)')
ax2.set_xlabel('\u03C61(x)')

ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(phi_l[:, 1], phi_l[:, 4])
ax3.set_ylabel('\u03C64(x)')
ax3.set_xlabel('\u03C61(x)')

ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(phi_l[:, 1], phi_l[:, 5])
ax4.set_ylabel('\u03C65(x)')
ax4.set_xlabel('\u03C61(x)')

ax5 = fig.add_subplot(gs[1, 2])
ax5.scatter(phi_l[:, 1], phi_l[:, 6])
ax5.set_ylabel('\u03C66(x)')
ax5.set_xlabel('\u03C61(x)')

ax6 = fig.add_subplot(gs[2, 0])
ax6.scatter(phi_l[:, 1], phi_l[:, 7])
ax6.set_ylabel('\u03C67(x)')
ax6.set_xlabel('\u03C61(x)')

ax7 = fig.add_subplot(gs[2, 1])
ax7.scatter(phi_l[:, 1], phi_l[:, 8])
ax7.set_ylabel('\u03C68(x)')
ax7.set_xlabel('\u03C61(x)')

ax8 = fig.add_subplot(gs[2, 2])
ax8.scatter(phi_l[:, 1], phi_l[:, 9])
ax8.set_ylabel('\u03C69(x)')
ax8.set_xlabel('\u03C61(x)')
plt.savefig("dif-3.png")
plt.show()import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
from matplotlib.gridspec import GridSpec



#function for calculating the eigenvectors, and eigenvalues in diffusion map
def diffusion_map(L,N,data):
    D = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
           D[i,j] =np.linalg.norm(data[i,:]-data[j,:])

    epsilon=np.amax(D)*0.05
    W=np.power(D,2)
    W= -1*W
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
    A, V = np.linalg.eig(T_hat)
    Al = A[0:L+1]
    Vl=V[:,:L+1]
    lambda_l = np.sqrt(np.power(Al,1/epsilon))
    phi_l=np.dot(Qpwr,Vl)
    return phi_l,lambda_l
N=1000
L=5
X = []
for i in range(N):
    X.append((math.cos(2*math.pi*i/(N+1)),math.sin(2*math.pi*i/(N+1))))
data = np.array(X)
phi_l,lambda_l = diffusion_map(L,N,data)


#calculating the values for each tk
Tk = []
for k in range(N):
    Tk.append(2*math.pi*k/(N+1))

#plotting the phi_l(x) against tk

fig = plt.figure(constrained_layout=True)
gs = GridSpec(3, 2, figure=fig)

ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(Tk,phi_l[:,0])
ax0.set_xlabel('x')
ax0.set_ylabel('\u03C60(x)')

ax1=fig.add_subplot(gs[0, 1])
ax1.plot(Tk,phi_l[:,1])
ax1.set_ylabel('\u03C61(x)')
ax1.set_xlabel('x')

ax2=fig.add_subplot(gs[1,0])
ax2.plot(Tk,phi_l[:,2])
ax2.set_ylabel('\u03C62(x)')
ax2.set_xlabel('x')

ax3=fig.add_subplot(gs[1,1])
ax3.plot(Tk,phi_l[:,3])
ax3.set_ylabel('\u03C63(x)')
ax3.set_xlabel('x')

ax4=fig.add_subplot(gs[2,0])
ax4.plot(Tk,phi_l[:,4])
ax4.set_ylabel('\u03C64(x)')
ax4.set_xlabel('x')

ax5=fig.add_subplot(gs[2,1])
ax5.plot(Tk,phi_l[:,5])
ax5.set_ylabel('\u03C65(x)')
ax5.set_xlabel('x')
plt.savefig('xx.png')
plt.show()

from matplotlib.pyplot import figure
from sklearn import datasets
sr = datasets.make_swiss_roll(n_samples=1500, noise=0.0, random_state=None)
Y = np.array(sr[0])
phi_l,lambda_l = diffusion_map(10,1500,Y)


#plotting the phi_l against phi_1 which is the first non-constant eigenvector
fig = plt.figure(figsize=(30, 20),constrained_layout=True)
gs = GridSpec(3, 3, figure=fig)
#figure(num=None, figsize=(8, 6), dpi=80)

ax0 = fig.add_subplot(gs[0,0])
ax0.scatter(phi_l[:,1],phi_l[:,0])
ax0.set_xlabel('\u03C60(x)')
ax0.set_ylabel('\u03C61(x)')

ax1=fig.add_subplot(gs[0,1])
ax1.scatter(phi_l[:,1],phi_l[:,2])
ax1.set_ylabel('\u03C62(x)')
ax1.set_xlabel('\u03C61(x)')

ax2=fig.add_subplot(gs[0,2])
ax2.scatter(phi_l[:,1],phi_l[:,3])
ax2.set_ylabel('\u03C63(x)')
ax2.set_xlabel('\u03C61(x)')

ax3=fig.add_subplot(gs[1,0])
ax3.scatter(phi_l[:,1],phi_l[:,4])
ax3.set_ylabel('\u03C64(x)')
ax3.set_xlabel('\u03C61(x)')

ax4=fig.add_subplot(gs[1,1])
ax4.scatter(phi_l[:,1],phi_l[:,5])
ax4.set_ylabel('\u03C65(x)')
ax4.set_xlabel('\u03C61(x)')

ax5=fig.add_subplot(gs[1,2])
ax5.scatter(phi_l[:,1],phi_l[:,6])
ax5.set_ylabel('\u03C66(x)')
ax5.set_xlabel('\u03C61(x)')

ax6=fig.add_subplot(gs[2,0])
ax6.scatter(phi_l[:,1],phi_l[:,7])
ax6.set_ylabel('\u03C67(x)')
ax6.set_xlabel('\u03C61(x)')

ax7=fig.add_subplot(gs[2,1])
ax7.scatter(phi_l[:,1],phi_l[:,8])
ax7.set_ylabel('\u03C68(x)')
ax7.set_xlabel('\u03C61(x)')

ax8=fig.add_subplot(gs[2,2])
ax8.scatter(phi_l[:,1],phi_l[:,9])
ax8.set_ylabel('\u03C69(x)')
ax8.set_xlabel('\u03C61(x)')
plt.savefig("dif-2.png")
plt.show()
Y -= Y.mean(axis=0)
U,S,Vh = np.linalg.svd(Y,full_matrices=False ,compute_uv=True)
#getting the direction of PCs
pc = Vh
#adding the magnitude to PCs by mutliplying by coresponding simgas
pc[0,:]*= S[0]
pc[1,:]*=S[1]
pc[2,:]*=S[2]
trace = np.sum(S)
e1 = S[0]/trace
e2 = S[1]/trace
e3 = S[2]/trace
print("The first PC: "+ str(pc[0,:]))
print("Energy of first PC: "+ str(e1))
print("The second  PC: "+ str(pc[1,:]))
print("Energy of second PC: "+ str(e2))
print("the third principle component: "+ str(pc[2,:]))
print("Energy of third PC: "+ str(e3))

import pandas as pd
df = pd.read_csv("data_DMAP_PCA_vadere.txt",header=None,delimiter=' ')

data= df.values

phi_l, lambda_l = diffusion_map(100,1000,data)

fig = plt.figure(figsize=(30, 20),constrained_layout=True)
gs = GridSpec(3, 3, figure=fig)
#figure(num=None, figsize=(8, 6), dpi=80)

ax0 = fig.add_subplot(gs[0,0])
ax0.scatter(phi_l[:,1],phi_l[:,0])
ax0.set_xlabel('\u03C61(x)')
ax0.set_ylabel('\u03C60(x)')

ax1=fig.add_subplot(gs[0,1])
ax1.scatter(phi_l[:,1],phi_l[:,2])
ax1.set_ylabel('\u03C62(x)')
ax1.set_xlabel('\u03C61(x)')

ax2=fig.add_subplot(gs[0,2])
ax2.scatter(phi_l[:,1],phi_l[:,3])
ax2.set_ylabel('\u03C63(x)')
ax2.set_xlabel('\u03C61(x)')

ax3=fig.add_subplot(gs[1,0])
ax3.scatter(phi_l[:,1],phi_l[:,4])
ax3.set_ylabel('\u03C64(x)')
ax3.set_xlabel('\u03C61(x)')

ax4=fig.add_subplot(gs[1,1])
ax4.scatter(phi_l[:,1],phi_l[:,5])
ax4.set_ylabel('\u03C65(x)')
ax4.set_xlabel('\u03C61(x)')

ax5=fig.add_subplot(gs[1,2])
ax5.scatter(phi_l[:,1],phi_l[:,6])
ax5.set_ylabel('\u03C66(x)')
ax5.set_xlabel('\u03C61(x)')

ax6=fig.add_subplot(gs[2,0])
ax6.scatter(phi_l[:,1],phi_l[:,7])
ax6.set_ylabel('\u03C67(x)')
ax6.set_xlabel('\u03C61(x)')

ax7=fig.add_subplot(gs[2,1])
ax7.scatter(phi_l[:,1],phi_l[:,8])
ax7.set_ylabel('\u03C68(x)')
ax7.set_xlabel('\u03C61(x)')

ax8=fig.add_subplot(gs[2,2])
ax8.scatter(phi_l[:,1],phi_l[:,9])
ax8.set_ylabel('\u03C69(x)')
ax8.set_xlabel('\u03C61(x)')
plt.savefig("dif-3.png")
plt.show()