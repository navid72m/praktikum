from PIL import Image, ImageDraw, ImageFont

if __name__ == '__main__':
    step_count = 25
    height = 600
    width = 600
    image = Image.new(mode='L', size=(height, width), color=255)

    # Draw some lines
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height
    step_size = int(image.width / step_count)

    for x in range(0, image.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=128)

    x_start = 0
    x_end = image.width

    for y in range(0, image.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=128)

    draw.rectangle((5*step_size,10*step_size , 6*step_size,11*step_size ), fill=134, outline=(134) ,width=step_size)
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
    draw.text((5*step_size,10*step_size), "P", font=fnt, fill=0)



    

    del draw

    image.show()
