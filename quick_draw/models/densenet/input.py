import random
import tensorflow as tf
from quick_draw.tfrecords.create_from_csv_files import decode_example


def _decode_example(serialized_example):
    features, labels = decode_example(serialized_example)
    features['drawing'] = tf.decode_raw(features['drawing'], tf.uint8)
    return features, labels


def _normalize(features, label):
    """Convert `image` from [0, 255] -> [0, 1] floats."""
    features['drawing'] = tf.cast(features['drawing'], tf.float32) / 255
    return features, label


def input_fn(file_paths, batch_size, only_recognized=False, epochs=None, shuffle=True):
    if shuffle:
        file_paths = list(file_paths)
        random.shuffle(file_paths)

    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(_decode_example)

    if only_recognized:
        dataset = dataset.filter(lambda features, label: tf.cast(features['recognized'], tf.bool))

    dataset = dataset.map(_normalize)

    if shuffle:
        dataset = dataset.shuffle(10000)

    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()
