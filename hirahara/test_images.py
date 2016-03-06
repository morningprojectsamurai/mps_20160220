from PIL import Image, ImageDraw


def rectangle(start_x, start_y, width, height, color=255, fill=255, canvas_width=256, canvas_height=256):
    image = Image.new('L', (canvas_width, canvas_height))
    draw = ImageDraw.Draw(image)
    draw.rectangle([start_x, start_y, start_x + width, start_y + height], fill=fill, outline=color)
    del draw
    return image


def ellipse(start_x, start_y, width, height, color=255, fill=255, canvas_width=256, canvas_height=256):
    image = Image.new('L', (canvas_width, canvas_height))
    draw = ImageDraw.Draw(image)
    draw.ellipse([start_x, start_y, start_x + width, start_y + height], fill=fill, outline=color)
    del draw
    return image
