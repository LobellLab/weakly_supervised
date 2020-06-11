import tensorflow as tf
import numpy as np


def parse_function(example, num_channels, img_size,
                   has_class_labels, has_seg_labels):
    """
    Transforms an example into a pair of a tensor and a float, representing
    an image and its label, respectively.
    """
    feature_dict = {
        'image': tf.FixedLenFeature((img_size, img_size, num_channels), tf.float32),
    }
    if has_class_labels:
        feature_dict.update({'label': tf.FixedLenFeature((1,), tf.float32)})
    if has_seg_labels:
        feature_dict.update(
            {'seg_labels': tf.FixedLenFeature((img_size, img_size), tf.float32)})
    parsed_features = tf.parse_single_example(example, features=feature_dict)

    ret = [parsed_features['image']]
    if has_class_labels:
        ret.append(parsed_features['label'])
    else:
        # TF doesn't accept None, we filter this out later anyway
        ret.append(0)

    if has_seg_labels:
        ret.append(parsed_features['seg_labels'])
    else:
        ret.append(0)
    return tuple(ret)


def augment_data(image, label, seg_label, perform_random_flip_and_rotate,
                 num_channels, has_seg_labels):
    """
    Image augmentation for training. Applies the following operations:
        - Horizontally flip the image with probabiliy 0.5
        - Vertically flip the image with probability 0.5
        - Apply random rotation
    """
    if perform_random_flip_and_rotate:
        if has_seg_labels:
            image = tf.concat([image, tf.expand_dims(seg_label, -1)], 2)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        rotate_angle = tf.random_shuffle([0.0, 90.0, 180.0, 270.0])[0]
        image = tf.contrib.image.rotate(
            image, rotate_angle * np.pi / 180.0, interpolation='BILINEAR')
        if has_seg_labels:
            seg_label = image[:, :, -1]

        image = image[:,:,:num_channels]

    return image, label, seg_label


def tfrecord_iterator(is_training, file_names, params):

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    def parse_fn(p):
        return parse_function(
            p, params.num_channels, params.img_size, params.has_class_labels,
            params.has_seg_labels)

    def train_fn(f, l, sl):
        return augment_data(
            f, l, sl, params.use_random_flip_and_rotate, params.num_channels,
            params.has_seg_labels)

    if is_training:
        dataset = (tf.data.TFRecordDataset(file_names)
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1))
    else:
        dataset = (tf.data.TFRecordDataset(file_names)
            .map(parse_fn)
            .batch(params.batch_size)
            .prefetch(1))

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels, seg_labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images,
              'iterator_init_op': iterator_init_op}
    if params.has_class_labels:
        inputs.update({'labels': labels})
    if params.has_seg_labels:
        inputs.update({'seg_labels': seg_labels})

    return inputs
