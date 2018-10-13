import csv
import json
from PIL import Image, ImageDraw
import numpy as np
from quick_draw.tfrecords.utils import encode_bitmap_example
from quick_draw.utils import project_dir
import tensorflow as tf


def convert_strokes_to_bitmap(strokes):
    image = Image.new('L', (256, 256), color=0)
    image_draw = ImageDraw.Draw(image)
    for stroke in strokes:
        for i in range(len(stroke[0]) - 1):
            image_draw.line([stroke[0][i], stroke[1][i], stroke[0][i + 1], stroke[1][i + 1]], fill=255, width=15)

    image = image.resize((28, 28), Image.ANTIALIAS)

    return np.array(image)


def create_tfrecords(csv_path, tfrecords_output_file):
    with tf.python_io.TFRecordWriter(tfrecords_output_file) as writer:
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)
            i = 0
            for _, _, drawing in reader:
                bitmap = convert_strokes_to_bitmap(json.loads(drawing))
                serialized_example = encode_bitmap_example(bitmap, -1)
                writer.write(serialized_example)

                i += 1
                if i % 1000 == 0:
                    print(i)


if __name__ == '__main__':
    create_tfrecords(project_dir('data/submission/test_simplified.csv'),
                     project_dir('data/submission/test_simplified.tfrecords'))
