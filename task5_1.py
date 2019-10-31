from PIL import Image, ImageDraw, ImageFont, ImageColor
from matplotlib import pyplot
import numpy
import cv2
import time
import math

class Panel:

    step_count = 50
    height = 600
    width = 600
    image = Image.new(mode='RGB', size=(height, width), color=(255,255,255))

    # Draw some lines
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height
    step_size = int(image.width / step_count)
    numRow=50
    numCol=50

    def __init__(self, peds, obstacles, target, rMax):


        self.peds = peds
        self.target = target
        self.obstacles = obstacles
        self.rMax = rMax
        self.movementSize = 1
        self.drawGrid()
        #the value is 1 if it the node is already visited 0 otherwise
        self.visit=numpy.zeros((self.numRow,self.numCol))

        #the distance of each cell from the target is kept in this list
        self.distances=numpy.zeros((self.numRow,self.numCol))
        #queue used in dijkstra's algorithm
        #the entities are a list of 3 values, each cell x,y, and distance
        self.q=[]

        #the states of cells is stored in the list, if it's an obstacle the value is 1
        self.states=numpy.zeros((self.numRow,self.numCol))
        for obstacle in self.obstacles:
            self.drawRec(obstacle[0], obstacle[1], "O", (0 , 255 , 0))
            self.states[obstacle[0]][obstacle[1]] = 1


        maxdist = self.numRow+self.numCol+10

        for i in range(self.numRow):
            for j in range(self.numCol):
                self.distances[i][j]=maxdist
        self.distances[target[0]][target[1]]=0
        self.q.append([target[0],target[1],0])



        for ped in peds:
            self.drawRec(ped[0], ped[1], "", 255)

        self.drawRec(target[0], target[1], "", (0,0,255,0) )
        self.dijkstra(target[0], target[1])
        cv2.imshow("show", numpy.array(self.image))
        cv2.waitKey(1000)
    def drawGrid(self):
        for x in range(0, self.image.width, self.step_size):
            line = ((x, self.y_start), (x, self.y_end))
            self.draw.line(line, fill=128)

        x_start = 0
        x_end = self.image.width

        for y in range(0, self.image.height, self.step_size):
            line = ((x_start, y), (x_end, y))
            self.draw.line(line, fill=128)

    def drawRec(self, X, Y, text, color):
        self.draw.rectangle((X*self.step_size,Y*self.step_size , (X+1)*self.step_size,(Y+1)*self.step_size ), fill=color, outline=(color) )
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
        # self.draw.text((X*self.step_size,Y*self.step_size), text, font=fnt, fill=0)

    def moveRect(self, X_new, Y_new, X_old, Y_old, flag):

        self.drawRec( X_new, Y_new, "P",134)
        # print(X_new)
        if X_new != X_old or Y_old != Y_new:
            self.drawRec(X_old,Y_old, "",(255,255,255))
        self.drawGrid()
        if flag :
            cv2.imshow("show", numpy.array(self.image))
        # cv2.waitKey(1000)


        # del self.draw

    def dist(self,p1,p2):
        return math.sqrt( abs(p1[0]-p2[0])**2 +  abs(p1[1]-p2[1])**2)

    def cost(self,p1,p2,rMax):

        d = self.dist(p1,p2)
        if d < rMax :
            return 10*math.exp(1 / (d**2 - rMax**2))
        else:
            return 0

    def dijkstra(self,tx,ty):

        # initdij(tx,ty)
        while len(self.q) > 0:


            mindist= self.numRow+self.numCol


            #finding the node with min dist in q
            for node in self.q:

               #node[0] contains x, node[1] contains y, and node[2] contains distance
                if node[2] < mindist:
                    mindist=node[2]
                    nextn = node


            self.q.remove(nextn)

            self.visit [nextn[0]][nextn[1]]=1
            dx=[0,0,1,-1]
            dy=[1,-1,0,0]

            for i in dx:
                for j in dy:
                    if i*j == 0:
                        ux = nextn[0] + i
                        uy = nextn[1] + j
                        if  ux>=0 and uy>=0 and ux < self.numRow and uy < self.numCol and self.states[ux][uy] == 0:

                            if self.visit[ux][uy] == 0 :

                                if(self.distances[ux][uy] > self.distances[nextn[0]][nextn[1]] + 1):
                                    self.distances[ux][uy] = self.distances[nextn[0]][nextn[1]] + 1
                                    self.q.append([ux,uy,self.distances[ux][uy]])

    #pCord is the list containg pedestrian coordinates,and tCord for target

    #moveUpdated is the updated version of moveToTarget function
    #p is a single pedestrain with [px,py]
    #t is target with [tx,ty]
    #peds is the list of all pedestrains
    def moveUpdated(self,p,tCord,peds, flag):

        #p is removed from the list of peds
        pedsTmp = peds.copy()
        pedsTmp.remove(p)
        pX=p[0]
        pY=p[1]

        #for each neighbor of p, in addition to its distance to target, a cost is added based on
        #its distance from other pedestrains
        d1 = self.distances[pX-1][pY]
        for ped in pedsTmp:
            if d1 != 0:
                d1 +=  10*self.cost([pX-1, pY], ped , self.rMax)

        d2 = self.distances[pX+1,pY]
        for ped in pedsTmp:
            if d2 != 0:
                d2 +=  10*self.cost([pX+1,pY], ped , self.rMax)

        d3 = self.distances[pX,pY-1]
        for ped in pedsTmp:
            if d3 != 0:
                d3 +=  10*self.cost([pX,pY-1], ped , self.rMax)

        d4 = self.distances[pX,pY+1]
        for ped in pedsTmp:
            if d4 != 0:
                d4 +=  10*self.cost([pX,pY+1], ped , self.rMax)

        #if one of the neibors are outside the grid, dn equals to sum of rows and columns
        #which is the max distance
        if pX-1 < 0:
            d1 = self.numRow + self.numCol + 10
        if  pY-1 < 0:
            d3 = self.numRow + self.numCol + 10
        if  pY+1 > self.numRow :
            d4 = self.numRow + self.numCol + 10
        if pX+1 > self.numCol:
            d2 = self.numRow + self.numCol + 10

        ans = [p[0],p[1]]
        minD = min([d1,d2,d3,d4])
        # print(minD)
        if minD != 0:
            if minD == d1:
                ans[0]= pX-1
            elif minD == d2:
                ans[0]= pX+1
            elif minD == d3:
                ans[1]=pY-1
            else:
                ans[1]=pY+1
        else:
            global t2
            global timePrinted
            if timePrinted==False:
                t2= time.time()
                print(t2-t1)
                timePrinted=True
        self.moveRect(ans[0],ans[1],pX,pY, flag)
        return ans

    #peds is a list of pedestrains with [x,y] , target has [x,y]
    #numStep is num of iterations, rmax is the threshhold for the cost function
    def updateState(self, peds , numStep):

        for i in range(1,numStep+1):
            #in each step for each pedestrian move function is called
            for j in range(0, len(peds)):
                #peds[j] coordinates get updated
                if i % self.movementSize ==0:
                    peds[j] = self.moveUpdated(peds[j],self.target,peds, True)
                else:
                    peds[j] = self.moveUpdated(peds[j],self.target,peds, False)
                # print("after")
                # print("j: %d cord %s",j,str(peds[j]))

            cv2.waitKey(1000)





peds = [[1, 10, 1]]
# peds = [[1, 1], [1, 5]]
target = [13, 13]
obstacles = []
global timePrinted
timePrinted=False
p = Panel(peds, obstacles, target, 1)

t1=time.time()
p.updateState(p.peds,50)
print("hello")
print(t2-t1)
# for i in range(0, 25):
#     p.Xpedestrian, p.Ypedestrian = p.moveToTarget([p.Xpedestrian, p.Ypedestrian], [p.Xtarget, p.Ytarget])

cv2.waitKey(0)
