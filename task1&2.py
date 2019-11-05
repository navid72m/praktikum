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

    def __init__(self, Xpedestrian, Ypedestrian, Xtarget, Ytarget, obstacles):


        self.Xpedestrian = Xpedestrian
        self.Ypedestrian = Ypedestrian
        self.Xtarget = Xtarget
        self.Ytarget = Ytarget
        self.obstacles = obstacles
        self.drawGrid()

        self.draw.rectangle((Xpedestrian*self.step_size,Ypedestrian*self.step_size , (Xpedestrian+1)*self.step_size,(Ypedestrian+1)*self.step_size ), fill=134, outline=(134) ,width=self.step_size)
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
        self.draw.text((Xpedestrian*self.step_size,Ypedestrian*self.step_size), "P", font=fnt, fill=0)
        self.states=numpy.zeros((self.numRow,self.numCol))
        for obstacle in self.obstacles:
            self.drawRec(obstacle[0], obstacle[1], "O", 134)
            self.states[obstacle[0]][obstacle[1]] = 1
        self.drawRec(Xtarget, Ytarget, "T", 134)
        # cv2.imshow("show", numpy.array(self.image))
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
        print(X_new)
        if X_new != X_old or Y_old != Y_new:
            self.drawRec(X_old,Y_old, "",255)
        self.drawGrid()
        cv2.imshow("show", numpy.array(self.image))
        cv2.waitKey(1000)


        # del self.draw
    #this method calculate the euclidean distance between p1 and p2
    def dist(self,p1,p2):
        return math.sqrt( abs(p1[0]-p2[0])**2 +  abs(p1[1]-p2[1])**2)

    def moveToTarget(self,pCord, tCord):

        pX=pCord[0]
        pY=pCord[1]
        Xnew = pCord[0]
        Ynew = pCord[1]
        d=-1
        d1 = self.dist([pX-1, pY] , tCord)
        d2 = self.dist([pX+1,pY], tCord)
        d3 = self.dist([pX,pY-1], tCord)
        d4 = self.dist([pX,pY+1], tCord)
        minD = min([d1,d2,d3,d4])
        print(minD)
        if minD != 0:
            if minD == d1:
                Xnew = pX-1
            elif minD == d2:
                Xnew = pX+1
            elif minD == d3:
                Ynew=pY-1
            else:
                Ynew=pY+1

        self.moveRect(Xnew, Ynew, pX, pY)
        return Xnew, Ynew



p = Panel(2, 23, 25, 25,[[2, 24],[ 3,24]])

for i in range(0, 25):
    p.Xpedestrian, p.Ypedestrian = p.moveToTarget([p.Xpedestrian, p.Ypedestrian], [p.Xtarget, p.Ytarget])

cv2.waitKey(0)
