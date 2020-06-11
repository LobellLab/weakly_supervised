import tensorflow as tf


def conv_conv_pool(X, f, stage, params, is_training=False, pool=True):

    conv_name_base = 'down' + str(stage)
    bn_name_base = 'bn' + str(stage)

    bn_momentum = params.bn_momentum
    l2_lambda = params.l2_lambda

    F1, F2 = f

    X = tf.layers.conv2d(
        inputs=X,
        filters=F1,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_lambda),
        name=conv_name_base + 'a')
    X = tf.layers.batch_normalization(
        inputs=X,
        axis=3,
        momentum=bn_momentum,
        training=is_training,
        name=bn_name_base + 'a')
    X = tf.nn.relu(X)

    X = tf.layers.conv2d(
        inputs=X,
        filters=F2,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_lambda),
        name=conv_name_base + 'b')
    X = tf.layers.batch_normalization(
        inputs=X,
        axis=3,
        momentum=bn_momentum,
        training=is_training,
        name=bn_name_base + 'b')
    X = tf.nn.relu(X)

    if not pool:
        return X

    pool = tf.layers.max_pooling2d(
        inputs=X,
        pool_size=(2,2),
        strides=(2,2),
        padding='valid')

    return X, pool


def upconv_concat(X, X_prev, f, stage, params):

    conv_name_base = 'up' + str(stage)
    l2_lambda = params.l2_lambda

    upconv = tf.layers.conv2d_transpose(
        inputs=X,
        filters=f,
        kernel_size=2,
        strides=2,
        padding='valid',
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_lambda),
        name=conv_name_base)

    prev_shape = tf.shape(X_prev)
    curr_shape = tf.shape(upconv)
    offsets = [0, (prev_shape[1] - curr_shape[1]) // 2, (prev_shape[2] - curr_shape[2]) // 2, 0]
    new_shape = [-1, curr_shape[1], curr_shape[2], -1]
    X_prev_cropped = tf.reshape(tf.slice(X_prev, offsets, new_shape), curr_shape)

    return tf.concat([upconv, X_prev_cropped], axis=-1)


def UNet(is_training, inputs, params):

    images = inputs['images']
    num_layers = params.num_layers
    starting_filters = params.starting_filters

    # U-Net convolutional layers
    convs = []
    conv, pool = conv_conv_pool(images, [starting_filters, starting_filters], 1, params, is_training)
    convs.append(conv)

    # add more convlutional layers in loop
    for l in range(2, num_layers):
        num_filters = starting_filters * 2**(l-1)
        conv, pool = conv_conv_pool(pool, [num_filters, num_filters], l, params, is_training)
        convs.append(conv)

    if num_layers == 2:
        num_filters = starting_filters

    # last convolutional layer (no pool)
    conv = conv_conv_pool(pool, [num_filters * 2, num_filters * 2], num_layers, params, is_training, pool=False)

    # convolution transpose layers (deconvolutional layers)
    for l in range(1, num_layers):
        num_filters = starting_filters * 2**(num_layers - l - 1)
        up = upconv_concat(conv, convs.pop(), num_filters, l, params)
        conv = conv_conv_pool(up, [num_filters, num_filters], l+num_layers, params, is_training, pool=False)

    CAM_input = tf.identity(conv)

    return CAM_input



def build_UNet(mode, inputs, params, reuse=False):
    """
    Function defines the graph operations.
    Input:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels, etc.)
        params: (params) contains hyperparameters of the model
        reuse: (bool) whether to reuse the weights
    Returns:
        model_spec: (dict) contains the graph operations for training/evaluation
    """
    is_training = (mode == 'train')
    labels = inputs.get('labels')
    seg_labels = inputs.get('seg_labels')

    with tf.variable_scope('model', reuse=reuse):
        CAM_input = UNet(is_training, inputs, params)

        if params.unet_type == 'sigmoid_gap':
            cam_unnorm = tf.layers.conv2d(
                inputs=CAM_input,
                filters=1,
                kernel_size=(1,1),
                strides=(1,1),
                padding='valid',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(params.l2_lambda),
                name='final_conv')

            cam_unnorm = tf.squeeze(cam_unnorm, 3) # get rid of last dim
            cam = tf.sigmoid(cam_unnorm)
            cam_predictions = tf.round(cam)
        
        elif params.unet_type == 'gap_sigmoid':
            # out = tf.sigmoid(CAM_input)
            cam = tf.reduce_mean(CAM_input, axis=[1,2])
            cam = tf.contrib.layers.flatten(cam)
            logits = tf.layers.dense(inputs=cam, units=params.num_labels, name='dense')


    if params.task_type == 'classification':

        if params.unet_type == 'sigmoid_gap':
            expits = tf.reduce_mean(cam, axis=[1, 2])
            expits = tf.expand_dims(expits, -1)

        elif params.unet_type == 'gap_sigmoid':
            expits = tf.sigmoid(logits)

        predictions = tf.round(expits)

    else:
        predictions = cam_predictions

    # possibly classification task also has segmentation labels
    if seg_labels is not None and params.unet_type == 'sigmoid_gap':
        # sizes might not match, cut the segmentation label down to size
        extra_h = tf.shape(seg_labels)[1] - tf.shape(cam)[1]
        extra_h_before = extra_h // 2
        extra_h_after = tf.shape(seg_labels)[1] - (extra_h - extra_h_before)
        extra_w = tf.shape(seg_labels)[2] - tf.shape(cam)[2]
        extra_w_before = extra_w // 2
        extra_w_after = tf.shape(seg_labels)[2] - (extra_w - extra_w_before)
        seg_labels_cut = seg_labels[:, extra_h_before:extra_h_after, extra_w_before:extra_w_after]
        # cam or cam unnorm
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=seg_labels_cut, logits=cam_unnorm)
        weighted_ce = (seg_labels_cut * ce) / params.class_1_segmentation_proportion + (1-seg_labels_cut) * ce
        segmentation_loss = tf.reduce_mean(weighted_ce)

    # Define loss and task accuracy
    if params.task_type == 'classification':
        train_labels = labels
        ce = -train_labels * tf.log(expits + 1.0e-8) - (1 - train_labels) * tf.log(1 - expits + 1.0e-8)
        weighted_ce = (train_labels * ce) / params.class_1_segmentation_proportion + \
            (1 - train_labels) * ce
        loss = tf.reduce_mean(weighted_ce)
        logits = -tf.log((1.0 / expits) - 1 + 1e-8)
    else:
        train_labels = seg_labels_cut
        loss = segmentation_loss

    l2_loss = tf.losses.get_regularization_loss()
    loss += l2_loss
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(train_labels), predictions), tf.float32))

    # Define training step that minimizes loss with Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

    # METRICS AND SUMMARIES
    with tf.variable_scope('metrics'):
        metrics = {
            'loss': tf.metrics.mean(loss),
            'accuracy': tf.metrics.accuracy(labels=tf.round(train_labels), predictions=predictions)
        }
        if params.task_type == 'classification' and seg_labels is not None and params.unet_type == 'sigmoid_gap':
            metrics.update({'segmentation_loss': tf.metrics.mean(segmentation_loss),
                            'segmentation_acc': tf.metrics.accuracy(labels=seg_labels_cut, predictions=cam_predictions)
                           })
    # Group the update ops for the tf.metrics
    # print(metrics.values())
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='metrics')
    metrics_init_op = tf.variables_initializer(metric_variables)

    summaries = []
    summaries.append(tf.summary.scalar('loss', loss))
    summaries.append(tf.summary.scalar('accuracy', accuracy))
    summaries.append(tf.summary.scalar('loss_MA', metrics['loss'][0]))
    summaries.append(tf.summary.scalar('accuracy_MA', metrics['accuracy'][0]))
    if params.task_type == 'classification' and seg_labels is not None and params.unet_type == 'sigmoid_gap':
        summaries.append(tf.summary.scalar('segmentation_loss', segmentation_loss))
        summaries.append(tf.summary.scalar('segmentation_loss_MA', metrics['segmentation_loss'][0]))
        summaries.append(tf.summary.scalar('segmentation_acc_MA', metrics['segmentation_acc'][0]))

    if params.dict.get('rgb_image') in {None, False}:
        image_rgb = 255.0 * 5 * inputs['images'][:,:,:,1:4][:,:,:,::-1]
        summaries.append(tf.summary.image('image', image_rgb))
    else:
        image_rgb = inputs['images']

    if params.unet_type == 'sigmoid_gap':
        paddings_h = tf.shape(image_rgb)[1] - tf.shape(cam)[1]
        paddings_h_before = paddings_h // 2
        paddings_h_after = paddings_h - paddings_h_before
        paddings_w = tf.shape(image_rgb)[2] - tf.shape(cam)[2]
        paddings_w_before = paddings_w // 2
        paddings_w_after = paddings_w - paddings_w_before
        paddings = tf.convert_to_tensor([[0, 0], [paddings_h_before, paddings_h_after], [paddings_w_before, paddings_w_after]])
        cam_padded = tf.pad(cam, paddings, 'CONSTANT', constant_values=0.5)
        cam_rgb = 255.0 * tf.tile(tf.expand_dims(cam_padded,-1), [1,1,1,3])

        if seg_labels is not None:
            # tile the segmentation, label and image
            label_rgb = 255.0 * tf.tile(tf.expand_dims(seg_labels,-1), [1,1,1,3])
            concated = tf.concat([image_rgb, label_rgb], axis=2)
        else:
            concated = image_rgb

        hard_cam_padded = tf.pad(cam_predictions, paddings, 'CONSTANT', constant_values=0.5)
        hard_cam_rgb = 255.0 * tf.tile(tf.expand_dims(hard_cam_padded, -1), [1, 1, 1, 3])
        concated = tf.concat([concated, cam_rgb, hard_cam_rgb], axis=2)
        summaries.append(tf.summary.image('concatenated_images', concated, 50))

    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec['predictions'] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['cam'] = cam
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge(summaries)
    
    if params.unet_type == 'gap_sigmoid':
        model_spec['CAM_input'] = CAM_input
        model_spec['dense_kernel'] = [v for v in tf.trainable_variables() if "dense/kernel" in v.name][0]
        model_spec['dense_bias'] = [v for v in tf.trainable_variables() if "dense/bias" in v.name][0]

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec    
