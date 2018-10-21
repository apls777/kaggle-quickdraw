import logging
import os
import tensorflow as tf
from quick_draw.utils import project_dir
from tensorflow.python.lib.io import file_io


logging.getLogger().setLevel(logging.DEBUG)
tf.logging.set_verbosity(tf.logging.INFO)


def train(model_fn, input_fn, params):
    model_dir = params['model_dir']
    input_dir = params['input_dir']
    batch_size = params['batch_size']
    save_checkpoints_secs = params.get('save_checkpoints_secs', 600)
    save_summary_steps = params.get('save_summary_steps', 500)
    only_recognized = params.get('only_recognized', False)
    eval_every_n_iter = params.get('eval_every_n_iter', 10000)
    eval_files_ids = params['eval_files']
    train_files_ids = params['train_files']

    logging.info('Model parameters: %s' % str(params))

    if not os.path.isabs(model_dir) and not model_dir.startswith('s3:'):
        model_dir = project_dir(model_dir)

    # create the model directory
    if not model_dir.startswith('s3:'):
        os.makedirs(model_dir, exist_ok=True)

    if not os.path.isabs(input_dir) and not input_dir.startswith('s3:'):
        input_dir = project_dir(input_dir)

    # get paths to training and evaluation tfrecords
    eval_files_ids = range(eval_files_ids[0], eval_files_ids[1] + 1)
    train_files_ids = range(train_files_ids[0], train_files_ids[1] + 1)
    eval_files = [os.path.join(input_dir, 'file_%d.tfrecords' % i) for i in eval_files_ids]
    train_files = [os.path.join(input_dir, 'file_%d.tfrecords' % i) for i in train_files_ids]

    logging.info('Number of eval files: %d' % len(eval_files))
    logging.info('Number of train files: %d' % len(train_files))

    # test access to training files
    file_io.stat(train_files[0])

    # create an estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(
            model_dir=model_dir,
            save_checkpoints_secs=save_checkpoints_secs,
            save_summary_steps=save_summary_steps,
        ),
        params=tf.contrib.training.HParams(**params),
    )

    # run evaluation every 10k steps
    eval_input_fn = lambda: input_fn(eval_files, batch_size, epochs=1, only_recognized=only_recognized)
    evaluator = tf.contrib.estimator.InMemoryEvaluatorHook(estimator,
                                                           input_fn=eval_input_fn,
                                                           every_n_iter=eval_every_n_iter)

    # train the model
    train_input_fn = lambda: input_fn(train_files, batch_size, only_recognized=only_recognized)
    estimator.train(input_fn=train_input_fn, hooks=[evaluator])


def predict(model_fn, input_fn, params, tfrecord_files):
    # get model directory
    model_dir = params['model_dir']
    if not os.path.isabs(model_dir) and not model_dir.startswith('s3:'):
        model_dir = project_dir(model_dir)

    batch_size = params['batch_size']

    # create an estimator
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                       params=tf.contrib.training.HParams(**params))

    # get predictions
    prediction_input_fn = lambda: input_fn(tfrecord_files, batch_size, epochs=1, shuffle=False)

    return estimator.predict(input_fn=prediction_input_fn)
