import logging
import tensorflow as tf


def model_fn(features, labels, mode, params):
    # input Layer
    input_shape = [-1, params.tfrecords_drawing_size, params.tfrecords_drawing_size, 1]
    input_layer = tf.reshape(features['drawing'], input_shape, name='input')

    log_tensor_shape(input_layer)

    # CNN layer
    conv1 = tf.layers.conv2d(input_layer, filters=64, kernel_size=(5, 5), padding='same', use_bias=False,
                             name='conv1')
    conv1_bn = tf.layers.batch_normalization(conv1, axis=3, epsilon=1.001e-5, name='conv1_bn')
    conv1_relu = tf.nn.relu(conv1_bn, name='relu1')
    conv1_pool = tf.layers.max_pooling2d(conv1_relu, pool_size=(2, 2), strides=(2, 2), name='pool1')

    log_tensor_shape(conv1_pool)

    blocks = params.blocks

    current_layer = conv1_pool
    for i, block in enumerate(blocks):
        # add convolution layers
        current_layer = dense_block(current_layer, block, name='conv%d' % (i + 2))
        log_tensor_shape(current_layer)

        if i + 1 < len(blocks):
            # add pooling layer
            current_layer = transition_block(current_layer, 0.5, name='pool%d' % (i + 2))
            log_tensor_shape(current_layer)

    conv_layers_bn = tf.layers.batch_normalization(current_layer, axis=3, epsilon=1.001e-5, name='bn')
    conv_layers_relu = tf.nn.relu(conv_layers_bn, name='relu')

    avg_pool = tf.reduce_mean(conv_layers_relu, axis=(1, 2), name='avg_pool')
    log_tensor_shape(avg_pool)

    # assignment_map = {}
    # for v in tf.trainable_variables():
    #     exclude_prefixes = [
    #         'pool3_',
    #         'conv4_',
    #         'pool4_',
    #         'conv5_',
    #         'bn/',
    #     ]
    #     exclude = False
    #     for prefix in exclude_prefixes:
    #         if v.name.startswith(prefix):
    #             exclude = True
    #             break
    #
    #     if not exclude:
    #         assignment_map[v.name[:-2]] = v
    #
    # tf.train.init_from_checkpoint(project_dir('training/densenet_b2s96w5_r'), assignment_map)

    # dense layer
    current_layer = avg_pool
    if params.dense_layer:
        dense = tf.layers.dense(inputs=current_layer, units=1024, activation=tf.nn.relu, name='dense1')
        current_layer = tf.layers.dropout(inputs=dense, rate=0.2, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                    name='dense1_dropout')
        log_tensor_shape(dense)

    # logits
    logits = tf.layers.dense(current_layer, units=params.num_classes, name='logits')
    log_tensor_shape(logits)

    predictions = {
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
        'class': tf.argmax(input=logits, axis=1),
        # 'top': tf.nn.top_k(logits)[1],
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
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['class']),
        'precision': tf.metrics.precision(labels=labels, predictions=predictions['class']),
        'recall': tf.metrics.recall(labels=labels, predictions=predictions['class']),
        # TODO:
        # 'precision_at_k': tf.metrics.precision_at_k(labels=tf.expand_dims(labels, axis=-1),
        #                                             predictions=predictions['probabilities'], k=3),
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


def log_tensor_shape(tensor):
    return logging.debug('"%s" shape: %s' % (tensor.name, str(tensor.get_shape())))
