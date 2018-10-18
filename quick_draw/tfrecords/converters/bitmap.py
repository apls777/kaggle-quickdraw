from PIL import Image, ImageDraw
import numpy as np
from quick_draw.tfrecords.converters.abstract import AbstractConverter


class BitmapConverter(AbstractConverter):
    def __init__(self, image_size=(28, 28), stroke_width=15):
        self.image_size = image_size
        self.stroke_width = stroke_width

    def convert(self, strokes: list):
        image = Image.new('L', (256, 256), color=0)
        image_draw = ImageDraw.Draw(image)
        for stroke in strokes:
            for i in range(len(stroke[0]) - 1):
                image_draw.line([stroke[0][i], stroke[1][i], stroke[0][i + 1], stroke[1][i + 1]], fill=255,
                                width=self.stroke_width)

        image = image.resize(self.image_size, Image.ANTIALIAS)

        return np.array(image)
