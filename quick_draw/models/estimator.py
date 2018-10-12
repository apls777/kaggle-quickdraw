import json
import logging
import os
from importlib import import_module
import tensorflow as tf
from quick_draw.models.input import iterator_get_next
from quick_draw.models.params import load_model_params
from quick_draw.utils import project_dir


logging.getLogger().setLevel(logging.DEBUG)
tf.logging.set_verbosity(tf.logging.INFO)


def train(model_name):
    params = load_model_params(model_name)

    # read the labels
    input_dir = params['input_dir']
    with open(project_dir(os.path.join(input_dir, 'labels.json'))) as f:
        labels_map = json.load(f)

    # create the model directory
    model_dir = project_dir(params['model_dir'])
    os.makedirs(model_dir, exist_ok=True)

    batch_size = params['batch_size']
    logging.info('Batch size: %d' % batch_size)

    # get paths to training and evaluation tfrecords
    eval_files_ids = range(params['eval_files'][0], params['eval_files'][1] + 1)
    train_files_ids = range(params['train_files'][0], params['train_files'][1] + 1)
    eval_files = [project_dir(os.path.join(input_dir, 'file_%d.tfrecords' % i)) for i in eval_files_ids]
    train_files = [project_dir(os.path.join(input_dir, 'file_%d.tfrecords' % i)) for i in train_files_ids]

    logging.info('Number of eval files: %d' % len(eval_files))
    logging.info('Number of train files: %d' % len(train_files))

    model_fn_params = params
    model_fn_params['num_classes'] = len(labels_map)

    # create an estimator
    model_fn = getattr(import_module('%s.%s.model' % (__package__, model_name)), 'model_fn')
    estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=tf.estimator.RunConfig(
                    model_dir=model_dir,
                    save_checkpoints_secs=params['save_checkpoints_secs'],
                    save_summary_steps=params['save_summary_steps'],
            ),
            params=tf.contrib.training.HParams(**model_fn_params),
    )

    # run evaluation every 10k steps
    evaluator = tf.contrib.estimator.InMemoryEvaluatorHook(estimator,
                                                           input_fn=lambda: iterator_get_next(eval_files, batch_size, 1),
                                                           every_n_iter=params['eval_every_n_iter'])

    # train the model
    estimator.train(input_fn=lambda: iterator_get_next(train_files, batch_size),
                    hooks=[evaluator])


def predict(model_name, labels_map_path, tfrecord_files):
    params = load_model_params(model_name)

    # read the labels
    with open(labels_map_path) as f:
        labels_map = json.load(f)

    model_dir = project_dir(params['model_dir'])
    batch_size = params['batch_size']

    model_fn_params = params
    model_fn_params['num_classes'] = len(labels_map)

    # create an estimator
    model_fn = getattr(import_module('%s.%s.model' % (__package__, model_name)), 'model_fn')
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                       params=tf.contrib.training.HParams(**model_fn_params))

    # get predictions
    return estimator.predict(input_fn=lambda: iterator_get_next(tfrecord_files, batch_size, 1, shuffle=False))
