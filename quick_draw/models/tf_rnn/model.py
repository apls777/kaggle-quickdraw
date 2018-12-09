import tensorflow as tf


def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    lengths = features['length']
    inputs = features['drawing']

    convolved, lengths = _add_conv_layers(inputs, lengths, params.num_conv, params.conv_len, params.batch_norm,
                                          params.dropout, is_training)
    final_state = _add_rnn_layers(convolved, lengths, params.cell_type, params.num_layers, params.num_nodes,
                                  params.dropout, is_training)
    logits = _add_fc_layers(final_state, params.num_classes)

    # compute predictions
    predictions = tf.argmax(logits, axis=1)

    # return specification for predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={
            'logits': logits, 
            'predictions': predictions,
        })

    # loss
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    # return specification for training
    if is_training:
        train_op = tf.contrib.layers.optimize_loss(
            loss=cross_entropy,
            global_step=tf.train.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer='Adam',
            clip_gradients=params.gradient_clipping_norm,
            summaries=['learning_rate', 'loss', 'gradients', 'gradient_norm'],
        )

        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, train_op=train_op)

    # return specification for evaluation
    return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, eval_metric_ops={
        'accuracy': tf.metrics.accuracy(labels, predictions),
    })


def _add_conv_layers(inputs, lengths, num_conv, conv_len, batch_norm, dropout, is_training):
    """Adds convolution layers."""
    convolved = inputs
    for i in range(len(num_conv)):
        convolved_input = convolved
        if batch_norm:
            convolved_input = tf.layers.batch_normalization(convolved_input, training=is_training)

        # add dropout layer if enabled and not first convolution layer.
        if i > 0 and dropout:
            convolved_input = tf.layers.dropout(convolved_input, rate=dropout, training=is_training)

        convolved = tf.layers.conv1d(convolved_input,
                                     filters=num_conv[i],
                                     kernel_size=conv_len[i],
                                     activation='relu',
                                     strides=1,
                                     padding='same',
                                     name='conv1d_%d' % i)

    return convolved, lengths


def _add_regular_rnn_layers(convolved, lengths, cell_type, num_layers, num_nodes, dropout):
    """Adds RNN layers."""
    if cell_type == 'lstm':
        cell = tf.nn.rnn_cell.BasicLSTMCell
    elif cell_type == 'block_lstm':
        cell = tf.contrib.rnn.LSTMBlockCell
    else:
        raise ValueError('Unknown cell type')

    cells_fw = [cell(num_nodes) for _ in range(num_layers)]
    cells_bw = [cell(num_nodes) for _ in range(num_layers)]
    if dropout > 0.0:
        cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
        cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_bw]

    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw,
                                                                   cells_bw=cells_bw,
                                                                   inputs=convolved,
                                                                   sequence_length=lengths,
                                                                   dtype=tf.float32,
                                                                   scope='rnn_classification')

    return outputs


def _add_cudnn_rnn_layers(convolved, num_layers, num_nodes, dropout, is_training):
    """Adds CUDNN LSTM layers."""
    # Convolutions output [B, L, Ch], while CudnnLSTM is time-major.
    convolved = tf.transpose(convolved, [1, 0, 2])
    lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=num_layers,
                                          num_units=num_nodes,
                                          dropout=dropout if is_training else 0.0,
                                          direction='bidirectional')
    outputs, _ = lstm(convolved)
    # Convert back from time-major outputs to batch-major outputs.
    outputs = tf.transpose(outputs, [1, 0, 2])

    return outputs


def _add_rnn_layers(convolved, lengths, cell_type, num_layers, num_nodes, dropout, is_training):
    """Adds recurrent neural network layers depending on the cell type."""
    if cell_type != 'cudnn_lstm':
        outputs = _add_regular_rnn_layers(convolved, lengths, cell_type, num_layers, num_nodes, dropout)
    else:
        outputs = _add_cudnn_rnn_layers(convolved, num_layers, num_nodes, dropout, is_training)
    # outputs is [batch_size, L, N] where L is the maximal sequence length and N
    # the number of nodes in the last layer.
    mask = tf.tile(
        tf.expand_dims(tf.sequence_mask(lengths, tf.shape(outputs)[1]), 2),
        [1, 1, tf.shape(outputs)[2]])
    zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
    outputs = tf.reduce_sum(zero_outside, axis=1)

    return outputs


def _add_fc_layers(final_state, num_classes):
    """Adds a fully connected layer."""
    return tf.layers.dense(final_state, num_classes)
