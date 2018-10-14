import csv
import json
from PIL import Image, ImageDraw
import numpy as np
from quick_draw.tfrecords.utils import encode_bitmap_example, encode_stroke3_example
from quick_draw.utils import project_dir, read_json, package_dir
import tensorflow as tf


def convert_strokes_to_bitmap(strokes):
    image = Image.new('L', (256, 256), color=0)
    image_draw = ImageDraw.Draw(image)
    for stroke in strokes:
        for i in range(len(stroke[0]) - 1):
            image_draw.line([stroke[0][i], stroke[1][i], stroke[0][i + 1], stroke[1][i + 1]], fill=255, width=15)

    image = image.resize((28, 28), Image.ANTIALIAS)

    return np.array(image)


def convert_strokes_to_stroke3(strokes):
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


def create_bitmaps_tfrecords(csv_path, tfrecords_output_file):
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


def create_stroke3_tfrecords(csv_path, tfrecords_output_file):
    country_codes = read_json(package_dir('data/countries.json'))
    labels = read_json(package_dir('data/labels.json'))

    with tf.python_io.TFRecordWriter(tfrecords_output_file) as writer:
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)
            i = 0
            for country_code, drawing, key_id, recognized, _, label in reader:
                assert recognized in ['True', 'False']

                country_id = country_codes[country_code]
                label_id = labels[label.replace(' ', '_')]
                strokes = convert_strokes_to_stroke3(json.loads(drawing))
                key_id = int(key_id)
                recognized = 1 if recognized == 'True' else 0

                serialized_example = encode_stroke3_example(strokes, label_id, country_id, recognized, key_id)
                writer.write(serialized_example)

                i += 1
                if i % 1000 == 0:
                    print(i)


if __name__ == '__main__':
    # create_bitmaps_tfrecords(project_dir('data/submission/test_simplified.csv'),
    #                          project_dir('data/submission/test_simplified.tfrecords'))
    create_stroke3_tfrecords(project_dir('data/kaggle_simplified/csv/airplane.csv'),
                             project_dir('data/kaggle_simplified/tfrecords/test.tfrecords'))
