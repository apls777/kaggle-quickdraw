from tensorflow.python.framework.errors_impl import OutOfRangeError
from quick_draw.models.densenet.input import iterator_get_next
import tensorflow as tf
from quick_draw.utils import project_dir


def read_tfrecords(tfrecords_path: str, drawing_dtype, limit: int = 0):
    batch_size = min(32, limit)

    it = iterator_get_next(tfrecords_path, batch_size, epochs=1, shuffle=False)

    features = {}
    labels = []
    with tf.Session() as sess:
        i = 0
        while True:
            if limit and i >= limit:
                break

            try:
                features_batch, labels_batch = sess.run(it)

                labels += list(labels_batch)

                for key, value in features_batch.items():
                    if key not in features:
                        features[key] = []
                    features[key] += list(value)
            except OutOfRangeError:
                break

            i += batch_size

            if i % 10000 == 0:
                print(i)

    if limit:
        for key, value in features.items():
            features[key] = features[key][:limit]

    return features, labels


def read_bitmaps(tfrecords_path: str, limit: int = 0):
    return read_tfrecords(tfrecords_path, tf.uint8, limit)


def read_stroke3(tfrecords_path: str, limit: int = 0):
    return read_tfrecords(tfrecords_path, tf.int16, limit)


def main():
    # features, labels = read_bitmaps(project_dir('data/kaggle_simplified/tfrecords/bitmaps/file_1.tfrecords'), limit=2)
    features, labels = read_stroke3(project_dir('data/kaggle_simplified/test_stroke3/file_1.tfrecords'), limit=2)
    print(features)
    print(labels)


if __name__ == '__main__':
    main()
