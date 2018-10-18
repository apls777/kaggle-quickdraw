from quick_draw.tfrecords.converters.abstract import AbstractConverter
import numpy as np


class Stroke3Converter(AbstractConverter):
    def __init__(self, image_size=(28, 28), stroke_width=15):
        self.image_size = image_size
        self.stroke_width = stroke_width

    def convert(self, strokes: list):
        total_length = sum([len(stroke[0]) for stroke in strokes])
        assert total_length > 0
        stroke3_arr = np.zeros((total_length, 3), dtype=np.int16)

        offset = 0
        for stroke in strokes:
            assert len(stroke[0]) == len(stroke[1])

            for i in [0, 1]:
                stroke3_arr[offset:(offset + len(stroke[0])), i] = stroke[i]

            offset += len(stroke[0])
            stroke3_arr[offset - 1, 2] = 1

        stroke3_arr[1:, 0:2] -= stroke3_arr[0:-1, 0:2]
        stroke3_arr = stroke3_arr[1:, :]

        return stroke3_arr
