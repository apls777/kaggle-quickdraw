import json
import logging
import os
from collections import OrderedDict
import tensorflow as tf
import numpy as np
from quick_draw.tfrecords import encode_bitmap_example
from quick_draw.utils import shuffle_arrays


logging.getLogger().setLevel(logging.DEBUG)


def create_tfrecord_files(bitmaps_path, output_path, num_files, delete_bitmaps=False):
    os.makedirs(output_path, exist_ok=True)

    filenames = os.listdir(bitmaps_path)
    labels_map = OrderedDict()
    for filename in filenames:
        bitmap_path = os.path.join(bitmaps_path, filename)

        logging.debug('Creating %d chunks from the "%s" file...' % (num_files, bitmap_path))

        # label
        drawing_type = filename.split('.')[0].replace(' ', '_')
        assert drawing_type not in labels_map
        label_id = len(labels_map)
        labels_map[drawing_type] = label_id

        logging.debug('Label ID: %d' % label_id)

        # bitmap set
        bitmap_set = np.load(bitmap_path)
        np.random.shuffle(bitmap_set)

        # split and save bitmaps by chunks
        chunk_size = int(len(bitmap_set) / num_files)

        logging.debug('Number of items: %d' % len(bitmap_set))
        logging.debug('Chunk size: %d' % chunk_size)

        offset = 0
        for i in range(num_files):
            chunk_path = os.path.join(output_path, 'tmp_l%d_f%d.npy' % (label_id, i + 1))
            next_offset = len(bitmap_set) if (i + 1 == num_files) else (offset + chunk_size)
            np.save(chunk_path, bitmap_set[offset:next_offset])
            offset = next_offset

        assert offset == len(bitmap_set)

        # delete original bitmaps
        if delete_bitmaps:
            os.unlink(bitmap_path)
            logging.debug('Original file with bitmaps was deleted')

    # save labels to a json file
    labels_map_path = os.path.join(output_path, 'labels.json')
    logging.debug('Writing labels map to "%s"...' % labels_map_path)
    with open(labels_map_path, 'w') as f:
        json.dump(labels_map, f, indent=4)

    logging.debug('Creating %d tfrecord files...' % num_files)

    for i in range(num_files):
        # reading i-th chunk of each label
        bitmaps_chunks = []
        tfrecord_labels = []
        for label_id in labels_map.values():
            chunk_path = os.path.join(output_path, 'tmp_l%d_f%d.npy' % (label_id, i + 1))
            bitmaps_chunk = np.load(chunk_path)
            bitmaps_chunks.append(bitmaps_chunk)
            tfrecord_labels += [label_id] * len(bitmaps_chunk)

            # remove the chuck file
            os.unlink(chunk_path)

        # concatenate chunks
        tfrecord_bitmaps = np.concatenate(bitmaps_chunks)

        # shuffle data and labels
        tfrecord_labels = np.array(tfrecord_labels)
        tfrecord_bitmaps, tfrecord_labels = shuffle_arrays(tfrecord_bitmaps, tfrecord_labels)

        # creating tfrecords
        tfrecords_path = os.path.join(output_path, 'file_%d.tfrecords' % (i + 1))

        logging.debug('Writing "%s" file with %d records...' % (tfrecords_path, len(tfrecord_bitmaps)))

        with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
            for j in range(0, len(tfrecord_bitmaps)):
                serialized_example = encode_bitmap_example(tfrecord_bitmaps[j], tfrecord_labels[j])
                writer.write(serialized_example)


def main():
    base_path = '../test_data/bitmaps'
    output_path = '../test_data/tfrecords'
    num_files = 10
    create_tfrecord_files(base_path, output_path, num_files)


if __name__ == '__main__':
    main()
