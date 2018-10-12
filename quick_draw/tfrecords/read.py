from tensorflow.python.framework.errors_impl import OutOfRangeError

from quick_draw.models.input import iterator_get_next
import tensorflow as tf


def read_bitmaps(tfrecords_path: str, limit: int = 0):
    batch_size = 500
    it = iterator_get_next(tfrecords_path, batch_size, 1, shuffle=False)

    images = []
    labels = []
    with tf.Session() as sess:
        i = 0
        while True:
            if limit and i >= limit:
                break

            try:
                images_batch, labels_batch = sess.run(it)
                images += list(images_batch)
                labels += list(labels_batch)
            except OutOfRangeError:
                break

            i += batch_size

            if i % 10000 == 0:
                print(i)

    if limit:
        images = images[:limit]
        labels = labels[:limit]

    return list(zip(images, labels))
