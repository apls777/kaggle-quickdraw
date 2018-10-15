import tensorflow as tf


SHUFFLE_SEED = 873277663


def encode_bitmap_example(bitmap, label_id):
    """Serializes an image and its label."""
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bitmap.tostring()])),
                'y': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_id])),
            }))

    return example.SerializeToString()


def encode_stroke3_example(stroke3_arr, label_id: int, country_id: int, recognized: int, key_id: int):
    """Serializes an image and its label."""
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'strokes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[stroke3_arr.tostring()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_id])),
                'country': tf.train.Feature(int64_list=tf.train.Int64List(value=[country_id])),
                'recognized': tf.train.Feature(int64_list=tf.train.Int64List(value=[recognized])),
                'key': tf.train.Feature(int64_list=tf.train.Int64List(value=[key_id])),
            }))

    return example.SerializeToString()


def decode_bitmap_example(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'x': tf.FixedLenFeature([], tf.string),
            'y': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['x'], tf.uint8)
    label = tf.cast(features['y'], tf.int32)

    return image, label


def decode_stroke3_example(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'strokes': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'country': tf.FixedLenFeature([], tf.int64),
            'recognized': tf.FixedLenFeature([], tf.int64),
            'key': tf.FixedLenFeature([], tf.int64),
        })

    features = {
        'strokes': tf.decode_raw(features['strokes'], tf.int16),
        'country': tf.cast(features['country'], tf.uint8),
        'recognized': tf.cast(features['recognized'], tf.uint8),
        'key': tf.cast(features['key'], tf.uint64),
    }

    # ['AU', 'CA', 'IE', 'NZ', 'GB', 'US']

    label = tf.cast(features['label'], tf.int32)

    return features, label
