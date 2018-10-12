import tensorflow as tf


def model_fn(features, labels, mode, params):
    # input Layer
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # CNN layer
    conv1 = tf.layers.conv2d(input_layer, filters=64, kernel_size=(5, 5), padding='same', use_bias=False,
                             name='conv1')
    conv1_bn = tf.layers.batch_normalization(conv1, axis=3, epsilon=1.001e-5, name='conv1_bn')
    conv1_relu = tf.nn.relu(conv1_bn, name='relu1')
    conv1_pool = tf.layers.max_pooling2d(conv1_relu, pool_size=(2, 2), strides=(2, 2), name='pool1')

    blocks = [6, 12, 24, 16]

    dense_block1 = dense_block(conv1_pool, blocks[0], name='conv2')
    transition_block1 = transition_block(dense_block1, 0.5, name='pool2')
    dense_block2 = dense_block(transition_block1, blocks[1], name='conv3')

    dense_block2_bn = tf.layers.batch_normalization(dense_block2, axis=3, epsilon=1.001e-5, name='bn')
    dense_block2_relu = tf.nn.relu(dense_block2_bn, name='relu')

    avg_pool = tf.reduce_mean(dense_block2_relu, axis=(1, 2), name='avg_pool')

    # dense layer
    # dense = tf.layers.dense(inputs=avg_pool, units=1024, activation=tf.nn.relu)
    # dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # logits
    logits = tf.layers.dense(avg_pool, units=params.num_classes, name='fc')

    predictions = {
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
        'class': tf.argmax(input=logits, axis=1),
    }

    # return specification for predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # return specification for training
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # return specification for evaluation
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes']),
        'precision': tf.metrics.precision(labels=labels, predictions=predictions['classes']),
        'recall': tf.metrics.recall(labels=labels, predictions=predictions['classes']),
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def dense_block(inputs, blocks, name):
    for i in range(blocks):
        inputs = conv_block(inputs, 32, name='%s_block%d' % (name, i + 1))

    return inputs


def conv_block(inputs, growth_rate, name):
    # batch normalization and activation for previous layer
    bn = tf.layers.batch_normalization(inputs, axis=3, epsilon=1.001e-5, name='%s_0_bn' % name)
    relu = tf.nn.relu(bn, name='%s_0_relu' % name)

    # convolution 1
    conv1 = tf.layers.conv2d(relu, filters=growth_rate * 4, kernel_size=(1, 1), use_bias=False, name='%s_1_conv' % name)
    conv1_bn = tf.layers.batch_normalization(conv1, axis=3, epsilon=1.001e-5, name='%s_1_bn' % name)
    conv1_relu = tf.nn.relu(conv1_bn, name='%s_1_relu' % name)

    # convolution 2
    conv2 = tf.layers.conv2d(conv1_relu, filters=growth_rate, kernel_size=(3, 3), padding='same', use_bias=False,
                             name='%s_2_conv' % name)

    # concatenation
    concat = tf.concat([inputs, conv2], axis=3, name='%s_concat' % name)

    return concat


def transition_block(inputs, reduction, name):
    bn = tf.layers.batch_normalization(inputs, axis=3, epsilon=1.001e-5, name='%s_bn' % name)
    relu = tf.nn.relu(bn, name='%s_relu' % name)
    conv = tf.layers.conv2d(relu, filters=int(relu.get_shape()[3].value * reduction), kernel_size=(1, 1), use_bias=False,
                            name='%s_conv' % name)
    avg_pool = tf.layers.average_pooling2d(conv, pool_size=(2, 2), strides=(2, 2), name='%s_pool' % name)

    return avg_pool
