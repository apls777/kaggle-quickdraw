import tensorflow as tf


def encode_bitmap_example(bitmap, label_id):
    """Serializes an image and its label."""
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bitmap.tostring()])),
                'y': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_id])),
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
