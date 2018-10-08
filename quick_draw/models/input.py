import tensorflow as tf
from quick_draw.tfrecords.utils import decode_bitmap_example


def _normalize(image, label):
    """Convert `image` from [0, 255] -> [0, 1] floats."""
    image = tf.cast(image, tf.float32) / 255
    return image, label


def iterator_get_next(file_paths, batch_size, epochs=None):
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = tf.data.TFRecordDataset(dataset)
    dataset = dataset.map(decode_bitmap_example)
    dataset = dataset.map(_normalize)
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()
