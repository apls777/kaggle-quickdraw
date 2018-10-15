import csv
import json
import logging
from random import Random
from PIL import Image, ImageDraw
import numpy as np
from quick_draw.tfrecords.utils import encode_bitmap_example, encode_stroke3_example, SHUFFLE_SEED
from quick_draw.utils import project_dir, read_json, package_dir
import tensorflow as tf
import os


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


def create_stroke3_tfrecords(csv_lines, tfrecords_output_file, skip_header=True):
    country_codes = read_json(package_dir('data/countries.json'))
    labels = read_json(package_dir('data/labels.json'))

    with tf.python_io.TFRecordWriter(tfrecords_output_file) as writer:
        reader = csv.reader(csv_lines)
        if skip_header:
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


def create_tfrecord_files(csv_dir, output_path, num_files, delete_csv_files=False):
    os.makedirs(output_path, exist_ok=True)

    labels = read_json(package_dir('data/labels.json'))
    existing_label_ids = []

    filenames = os.listdir(csv_dir)
    for filename in filenames:
        csv_path = os.path.join(csv_dir, filename)

        logging.debug('Creating %d chunks from the "%s" file...' % (num_files, csv_path))

        # label
        label = filename.split('.')[0].replace(' ', '_')
        label_id = labels[label]
        existing_label_ids.append(label_id)
        logging.debug('Label: %s' % label)

        # read and shuffle the data
        with open(csv_path) as f:
            f.readline()
            lines = f.readlines()

        Random(SHUFFLE_SEED).shuffle(lines)

        # split and save lines by chunks
        chunk_size = len(lines) // num_files
        spread_size = len(lines) % num_files

        logging.debug('Number of items: %d' % len(lines))
        logging.debug('Chunk size: %d' % chunk_size)

        offset = 0
        for i in range(num_files):
            s = 1 if spread_size > i else 0
            chunk_path = os.path.join(output_path, 'tmp_l%d_f%d.csv' % (label_id, i + 1))
            next_offset = offset + chunk_size + s

            with open(chunk_path, 'w') as f:
                f.writelines(lines[offset:next_offset])

            offset = next_offset

        assert offset == len(lines)

        # delete original files
        if delete_csv_files:
            os.unlink(csv_path)
            logging.debug('File "%s" was deleted' % csv_path)

    logging.debug('Creating %d tfrecord files...' % num_files)

    for i in range(num_files):
        # reading i-th chunk of each label
        lines = []
        for label_id in existing_label_ids:
            chunk_path = os.path.join(output_path, 'tmp_l%d_f%d.csv' % (label_id, i + 1))

            with open(chunk_path) as f:
                chunk_lines = f.readlines()

            lines += chunk_lines

            # remove the chuck file
            os.unlink(chunk_path)

        # shuffle data
        Random(SHUFFLE_SEED).shuffle(lines)

        # create TFRecords file
        tfrecords_path = os.path.join(output_path, 'file_%d.tfrecords' % (i + 1))
        logging.debug('Writing "%s" file with %d records...' % (tfrecords_path, len(lines)))
        create_stroke3_tfrecords(lines, tfrecords_path, skip_header=False)

    if len(labels) != len(existing_label_ids):
        logging.warning('CSVs for some labels didn\'t exist. Existing labels IDs: %s.' % str(existing_label_ids))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    # create_bitmaps_tfrecords(project_dir('data/submission/test_simplified.csv'),
    #                          project_dir('data/submission/test_simplified.tfrecords'))

    # with open(project_dir('data/kaggle_simplified/csv/airplane.csv')) as f:
    #     create_stroke3_tfrecords(f, project_dir('data/kaggle_simplified/tfrecords/test.tfrecords'))

    create_tfrecord_files(project_dir('data/kaggle_simplified/csv'), project_dir('data/kaggle_simplified/tfrecords'), 1)
