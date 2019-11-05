from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot
import numpy
import cv2
import time
import math

class Panel:

    step_count = 26
    height = 600
    width = 600
    image = Image.new(mode='L', size=(height, width), color=255)

    # Draw some lines
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height
    step_size = int(image.width / step_count)
    numRow=25
    numCol=25

    def __init__(self, peds, target, rMax):


        self.peds = peds
        self.target = target
        self.rMax = rMax
        self.drawGrid()

        for ped in peds:
            self.drawRec(ped[0], ped[1], "P", 134)

        self.drawRec(target[0], target[1], "T", 134)
        cv2.imshow("show", numpy.array(self.image))
        cv2.waitKey(1000)
    #this method draw horizental and vertical lines and form a grid
    def drawGrid(self):
        for x in range(0, self.image.width, self.step_size):
            line = ((x, self.y_start), (x, self.y_end))
            self.draw.line(line, fill=128)

        x_start = 0
        x_end = self.image.width

        for y in range(0, self.image.height, self.step_size):
            line = ((x_start, y), (x_end, y))
            self.draw.line(line, fill=128)
    #this method takes position of the rectangle and text to be written in that cell and color is the color filled in the cell
    def drawRec(self, X, Y, text, color):
        self.draw.rectangle((X*self.step_size,Y*self.step_size , (X+1)*self.step_size,(Y+1)*self.step_size ), fill=color, outline=(color) ,width=self.step_size)
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
        self.draw.text((X*self.step_size,Y*self.step_size), text, font=fnt, fill=0)
    #this method takes new position and old position as arguments and move the pedestrian to the new position
    def moveRect(self, X_new, Y_new, X_old, Y_old):

        self.drawRec( X_new, Y_new, "P",134)
        # print(X_new)
        if X_new != X_old or Y_old != Y_new:
            self.drawRec(X_old,Y_old, "",255)
        self.drawGrid()
        cv2.imshow("show", numpy.array(self.image))
        cv2.waitKey(1000)


        # del self.draw
    #this method calculate the euclidean distance between p1 and p2
    def dist(self,p1,p2):
        return math.sqrt( abs(p1[0]-p2[0])**2 +  abs(p1[1]-p2[1])**2)
    #this method takes p1 and p2 and if the distance is less than threshhold rMax then return the cost
    def cost(self,p1,p2,rMax):

        d = self.dist(p1,p2)
        if d < rMax :
            return 10*math.exp(1 / (d**2 - rMax**2))
        else:
            return 0

    #pCord is the list containg pedestrian coordinates,and tCord for target

    #moveUpdated is the updated version of moveToTarget function
    #p is a single pedestrain with [px,py]
    #t is target with [tx,ty]
    #peds is the list of all pedestrains
    def moveUpdated(self,p,tCord,peds):

        #p is removed from the list of peds
        pedsTmp = peds.copy()
        pedsTmp.remove(p)
        pX=p[0]
        pY=p[1]

        #for each neighbor of p, in addition to its distance to target, a cost is added based on
        #its distance from other pedestrains
        d1 = self.dist([pX-1, pY] , tCord)
        for ped in pedsTmp:
            if d1 != 0:
                d1 +=  self.cost([pX-1, pY], ped , self.rMax)
                print("d1"+str(d1))

        d2 = self.dist([pX+1,pY], tCord)
        for ped in pedsTmp:
            if d2 != 0:
                d2 +=  self.cost([pX+1,pY], ped , self.rMax)
                print(d2)
        d3 = self.dist([pX,pY-1], tCord)
        for ped in pedsTmp:
            if d3 != 0:
                d3 +=  self.cost([pX,pY-1], ped , self.rMax)
                print(d3)
        d4 = self.dist([pX,pY+1], tCord)
        for ped in pedsTmp:
            if d4 != 0:
                d4 +=  self.cost([pX,pY+1], ped , self.rMax)
                print(d4)
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
        self.moveRect(ans[0],ans[1],pX,pY)
        return ans

    #peds is a list of pedestrains with [x,y] , target has [x,y]
    #numStep is num of iterations, rmax is the threshhold for the cost function
    def updateState(self, peds , numStep):

        for i in range(0,numStep):
            #in each step for each pedestrian move function is called
            for j in range(0, len(peds)):
                #peds[j] coordinates get updated
                peds[j] = self.moveUpdated(peds[j],self.target,peds)
                print("after")
                print("j: %d cord %s",j,str(peds[j]))





peds = [[1, 1], [1, 3], [1, 23], [23, 1], [23, 10]]
# peds = [[1, 1], [1, 5]]
target = [12, 13]
p = Panel(peds,target, 1)
p.updateState(p.peds,100)

# for i in range(0, 25):
#     p.Xpedestrian, p.Ypedestrian = p.moveToTarget([p.Xpedestrian, p.Ypedestrian], [p.Xtarget, p.Ytarget])

cv2.waitKey(0)
