import csv
import json
import logging
from random import Random
from quick_draw.tfrecords.converters.abstract import AbstractConverter
from quick_draw.tfrecords.converters.bitmap import BitmapConverter
from quick_draw.utils import project_dir, read_json, package_dir
import tensorflow as tf
import os


SHUFFLE_SEED = 873277663


def create_tfrecords(csv_dir, output_path, num_files, converter: AbstractConverter, shuffle_seed=SHUFFLE_SEED,
                     delete_csv_files=False):
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

        Random(shuffle_seed).shuffle(lines)

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
        Random(shuffle_seed).shuffle(lines)

        # create TFRecords file
        tfrecords_path = os.path.join(output_path, 'file_%d.tfrecords' % (i + 1))
        logging.debug('Writing "%s" file with %d records...' % (tfrecords_path, len(lines)))
        create_tfrecords_from_rows(lines, tfrecords_path, converter, skip_header=False)

    if len(labels) != len(existing_label_ids):
        logging.warning('CSVs for some labels didn\'t exist. Existing labels IDs: %s.' % str(existing_label_ids))


def create_tfrecords_from_rows(csv_rows, tfrecords_output_file, converter: AbstractConverter, skip_header=False):
    country_codes = read_json(package_dir('data/countries.json'))
    labels = read_json(package_dir('data/labels.json'))

    with tf.python_io.TFRecordWriter(tfrecords_output_file) as writer:
        reader = csv.reader(csv_rows)
        if skip_header:
            next(reader)

        i = 0
        for country_code, strokes_json, key_id, recognized, _, label in reader:
            assert recognized in ['True', 'False']

            country_id = country_codes[country_code]
            label_id = labels[label.replace(' ', '_')]
            key_id = int(key_id)
            recognized = 1 if recognized == 'True' else 0

            drawing = converter.convert(json.loads(strokes_json))

            serialized_example = encode_example(drawing, label_id, country_id, recognized, key_id)
            writer.write(serialized_example)

            i += 1
            if i % 1000 == 0:
                logging.debug(i)


def create_tf_records_from_file(file_path, tfrecords_output_file, converter: AbstractConverter, limit=0):
    with open(file_path) as f:
        return create_tfrecords_from_rows(f, tfrecords_output_file, converter, skip_header=True)


def encode_example(drawing, label_id: int, country_id: int, recognized: int, key_id: int):
    """Serializes an image and its label."""
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'drawing': tf.train.Feature(bytes_list=tf.train.BytesList(value=[drawing.tostring()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_id])),
                'country': tf.train.Feature(int64_list=tf.train.Int64List(value=[country_id])),
                'recognized': tf.train.Feature(int64_list=tf.train.Int64List(value=[recognized])),
                'key': tf.train.Feature(int64_list=tf.train.Int64List(value=[key_id])),
            }))

    return example.SerializeToString()


def decode_example(serialized_example, drawing_dtype):
    """Parses an image and label from the given `serialized_example`."""
    example = tf.parse_single_example(
        serialized_example,
        features={
            'drawing': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'country': tf.FixedLenFeature([], tf.int64),
            'recognized': tf.FixedLenFeature([], tf.int64),
            'key': tf.FixedLenFeature([], tf.int64),
        })

    features = {
        'drawing': tf.decode_raw(example['drawing'], drawing_dtype),
        'country': tf.cast(example['country'], tf.uint8),
        'recognized': tf.cast(example['recognized'], tf.uint8),
        'key': tf.cast(example['key'], tf.uint64),
    }

    label = tf.cast(example['label'], tf.int64)

    return features, label


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    # with open(project_dir('data/kaggle_simplified/csv/airplane.csv')) as f:
    #     create_stroke3_tfrecords(f, project_dir('data/kaggle_simplified/tfrecords/test.tfrecords'))

    bitmap_converter = BitmapConverter(image_size=(96, 96), stroke_width=5)
    print(len(bitmap_converter.convert([]).tobytes()))
    exit()

    create_tfrecords(project_dir('data/kaggle_simplified/csv'), project_dir('data/kaggle_simplified/tfrecords/bitmaps'),
                     num_files=1, converter=bitmap_converter)
