import logging
import tensorflow as tf


logging.getLogger().setLevel(logging.DEBUG)
tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features, labels, mode, params):
    """Model function.
    Args:
        features: image vector
        labels: sparse labels
        mode: one of tf.estimator.ModeKeys.{TRAIN, PREDICT, EVAL}
        params: a parameter dictionary with the following keys: num_layers,
            num_nodes, batch_size, num_conv, conv_len, num_classes, learning_rate.

    Returns:
        ModelFnOps for Estimator API.
    """
    # input Layer
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # CNN layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)  # [-1, 28, 28, 32]

    # pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)  # [-1, 14, 14, 32]

    # CNN layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)  # [-1, 14, 14, 64]

    # pooling layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)  # [-1, 7, 7, 64]

    # flat
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])  # [-1, 7 * 7 * 64]

    # dense layer
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)  # [-1, 1024]
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # logits layer
    logits = tf.layers.dense(inputs=dropout, units=params.num_classes)  # [-1, num_classes]

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        'classes': tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes']),
        'precision': tf.metrics.precision(labels=labels, predictions=predictions['classes']),
        'recall': tf.metrics.recall(labels=labels, predictions=predictions['classes']),
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
