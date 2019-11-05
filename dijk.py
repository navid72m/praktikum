
# coding: utf-8

# In[30]:


import numpy

nrow=10
ncol=10


#the value is 1 if it the node is already visited 0 otherwise
visit=numpy.zeros((nrow,ncol))

#the distance of each cell from the target is kept in this list
distances=numpy.zeros((nrow,ncol))
#queue used in dijkstra's algorithm
#the entities are a list of 3 values, each cell x,y, and distance
q=[]

#the states of cells is stored in the list, if it's an obstacle the value is 1
states=numpy.zeros((nrow,ncol))
states[1][1]=1

tx=1
ty=1

#used to initilize the queue and the distances
def initdij(tx,ty):

    maxdist = nrow+ncol+10

    for i in range(nrow):
        for j in range(ncol):
            distances[i][j]=maxdist
    distances[tx][ty]=0
    q.append([tx,ty,0])

def dijkstra(tx,ty):

    initdij(tx,ty)
    while len(q) > 0:

        print(distances)
        mindist= nrow+ncol


        #finding the node with min dist in q
        for node in q:

           #node[0] contains x, node[1] contains y, and node[2] contains distance
            if node[2] < mindist:
                mindist=node[2]
                nextn = node


        q.remove(nextn)
        print(q)
        visit [nextn[0]][nextn[1]]=1
        dx=[0,0,1,-1]
        dy=[1,-1,0,0]

        for i in dx:
            for j in dy:
                if i*j == 0:
                    ux = nextn[0] + i
                    uy = nextn[1] + j
                    if  ux>=0 and uy>=0 and ux < nrow and uy < ncol and states[ux][uy] == 0:

                        if visit[ux][uy] == 0 :

                            if(distances[ux][uy] > distances[nextn[0]][nextn[1]] + 1):
                                distances[ux][uy] = distances[nextn[0]][nextn[1]] + 1
                                q.append([ux,uy,distances[ux][uy]])


print(distances)
dijkstra(1,2)
print(distances)
