from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot
import numpy
import cv2
import time

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

    def __init__(self, Xpedestrian, Ypedestrian, Xtarget, Ytarget):


        self.Xpedestrian = Xpedestrian
        self.Ypedestrian = Ypedestrian
        self.drawGrid()

        self.draw.rectangle((Xpedestrian*self.step_size,Ypedestrian*self.step_size , (Xpedestrian+1)*self.step_size,(Ypedestrian+1)*self.step_size ), fill=134, outline=(134) ,width=self.step_size)
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
        self.draw.text((Xpedestrian*self.step_size,Ypedestrian*self.step_size), "T", font=fnt, fill=0)
        # cv2.imshow("show", numpy.array(self.image))

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
        self.draw.rectangle((X*self.step_size,Y*self.step_size , (X+1)*self.step_size,(Y+1)*self.step_size ), fill=color, outline=(color) ,width=self.step_size)
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
        self.draw.text((X*self.step_size,Y*self.step_size), text, font=fnt, fill=0)

    def moveRect(self, count):
        count = 5
        while (count != 0):
            self.drawRec(self.Xpedestrian + count,self.Ypedestrian, "P",134)
            print(self.Xpedestrian)
            cv2.imshow("show", numpy.array(self.image))
            cv2.waitKey(1000)
            count = count -1
            self.drawRec(self.Xpedestrian + count+1,self.Ypedestrian, "",255)
            self.drawGrid()

        # del self.draw



p = Panel(4, 24, 25, 25)
p.moveRect(5)
cv2.waitKey(0)
