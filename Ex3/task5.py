import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

postvis1 = open("/home/navid/Ex3/Ex3/output/corrupt/3.6/postvis.trajectories", "r")

fig = plt.figure(figsize=(7, 9))
gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 1])
ax0 = fig.add_subplot(gs[:, :])

index=0
Xarray = []
Yarray= []
Timearray =[]
desiredID = 2
for line in postvis1:
    if index != 0:
        token = line.split(" ")
        time = int(token[0])
        agentID = int(token[1])
        X = float(token[2])
        Y = float(token[3])
        if agentID == desiredID :
            Xarray.append(X)
            Yarray.append(Y)
            Timearray.append(time)



    index+=1
postvis1.close()
print(len(Xarray))
ax0.plot(Timearray,Xarray,'-', alpha=1)
ax0.plot(Timearray,Yarray,'-', alpha=1)


plt.tight_layout()
plt.show()