import random
import tensorflow as tf
from quick_draw.tfrecords.create_from_csv_files import decode_example


def _decode_example(serialized_example):
    features, labels = decode_example(serialized_example)
    features['drawing'] = tf.reshape(tf.decode_raw(features['drawing'], tf.int16), [-1, 3])
    features['length'] = tf.shape(features['drawing'])[0] // 3
    return features, labels


def _normalize(features, label):
    """TODO: std dev normalization"""
    features['drawing'] = features['drawing'] / 255
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
    # dataset = dataset.prefetch(50000)

    dataset = dataset.padded_batch(batch_size, padded_shapes=dataset.output_shapes)

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()
