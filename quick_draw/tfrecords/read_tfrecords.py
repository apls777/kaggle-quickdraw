from tensorflow.python.framework.errors_impl import OutOfRangeError
from quick_draw.models.input import iterator_get_next
import tensorflow as tf
from quick_draw.utils import project_dir


def read_bitmaps(tfrecords_path: str, drawing_dtype, limit: int = 0):
    batch_size = 32
    it = iterator_get_next(tfrecords_path, batch_size, epochs=1, shuffle=False, drawing_dtype=drawing_dtype)

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


def main():
    features, labels = read_bitmaps(project_dir('data/kaggle_simplified/tfrecords/bitmaps/file_1.tfrecords'),
                                    drawing_dtype=tf.uint8,
                                    limit=2)
    print(features)


if __name__ == '__main__':
    main()
