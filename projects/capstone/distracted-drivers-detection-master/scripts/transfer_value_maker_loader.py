import os

import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.contrib import slim


from preprocessing.preprocessing_factory import get_preprocessing
from nets import nets_factory

from checkpoints_downloader import ckpt_maker
from dataset_preparation import get_split, load_batch


####################################################
###            features types                    ###
####################################################

def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def float_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


####################################################
###       output layer and exclude scope         ###
####################################################

# give the scope of the layer before fully-connected layer or logits
def base_layer(MODEL):
    nets_base_map = {'vgg_16': 'vgg_16/pool5',
                     'vgg_19': 'vgg_19/pool5',
                     'inception_v1': 'Mixed_5c',
                     'inception_v2': 'Mixed_5c',
                     'inception_v3': 'Mixed_7c',
                     'inception_v4': 'Mixed_7d',
                     'inception_resnet_v2': 'Conv2d_7b_1x1',
                     'resnet_v2_50': 'resnet_v2_50/block4',
                     'resnet_v2_101': 'resnet_v2_101/block4',
                     'resnet_v2_152': 'resnet_v2_152/block4',
                     }
    return nets_base_map[MODEL]

# exclude scope of checkpoints
def exclude_scope(MODEL):
    nets_exclude_map = {'vgg_16': ['vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/fc8'],
                        'vgg_19': ['vgg_19/fc6', 'vgg_19/fc7', 'vgg_19/fc8'],
                        'inception_v1': ['InceptionV1/AuxLogits', 'InceptionV1/Logits'],
                        'inception_v2': ['InceptionV2/AuxLogits', 'InceptionV2/Logits'],
                        'inception_v3': ['InceptionV3/AuxLogits', 'InceptionV3/Logits'],
                        'inception_v4': ['InceptionV4/AuxLogits', 'InceptionV4/Logits'],
                        'inception_resnet_v2': ['InceptionResnetV2/AuxLogits', 'InceptionResnetV2/Logits'],
                        'resnet_v2_50': ['resnet_v2_50/logits'],
                        'resnet_v2_101': ['resnet_v2_101/logits'],
                        'resnet_v2_152': ['resnet_v2_152/logits'],
                        }
    return nets_exclude_map[MODEL]

####################################################
###      write & read shape of transfer-value    ###
####################################################

def write_shape_file(shape, MODEL, dataset_dir):
    '''Write a file with the shape of the output of MODEL base.
    
    Args: 
        shape(tuple): (height, width, channel)
        MODEL(str): model name 
        dataset_dir(str): file directory
    '''
    filename = 'shape_' + MODEL + '.txt'
    file_path = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(file_path, 'w') as f:
        f.write('{}, {}, {}'.format(shape[0], shape[1], shape[2]))
        
        
def read_shape_file(MODEL, dataset_dir):
    '''Read the shape file of MODEL.
    
    Args:
        MODEL(str): model name
        dataset_dir(str): file directory
        
    Return:
        shape(list): [height, width, channel]
    '''
    filename = 'shape_' + MODEL + '.txt'
    file_path = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(file_path, 'rb') as f:
        lines = f.read().decode()
        return list(map(int, lines.split(',')))

####################################################
###            array features and writer         ###
####################################################
    
def array_to_tfexample(array_b, class_id, img_b):
    return tf.train.Example(features=tf.train.Features(feature={
        'array/encoded': bytes_feature(array_b),
        'array/filename': bytes_feature(img_b),
        'array/class/label': int64_feature(class_id)}))

def array_to_tfrecord(array, class_id, img, writer):
    array = array.astype(np.float32)
    array_b = array.tostring()
    img_b = img.encode()
    example = array_to_tfexample(array_b, class_id, img_b)
    writer.write(example.SerializeToString())
    
####################################################
###            transfer-value maker             ###
####################################################

def tv_maker(MODEL, split_name, dataset_dir, file_pattern, tf_dir, checkpoint_file, log_dir):
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)

        # load model
        model = nets_factory.get_network_fn(MODEL, num_classes=10, is_training=False)
        # get image size
        image_size = model.default_image_size


        dataset = get_split(split_name=split_name, 
                            dataset_dir=dataset_dir, 
                            file_pattern=file_pattern, 
                            file_pattern_for_counting='drivers', 
                            items_to_descriptions=None)
        images, _, labels, image_names = load_batch(
            dataset=dataset, 
            batch_size=1, 
            MODEL=MODEL, 
            height=image_size, 
            width=image_size, 
            is_training=False)

        _, endpoints = model(images)

        base_scope = base_layer(MODEL)

        net = endpoints[base_scope]
        shape = net.get_shape()[1:]
        write_shape_file(shape, MODEL, tf_dir)

        exclusion = exclude_scope(MODEL)
        variables_to_restore = slim.get_variables_to_restore(exclude=exclusion)
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        sv = tf.train.Supervisor(logdir=log_dir, summary_op=None, saver=None, init_fn=restore_fn)

        with sv.managed_session() as sess:
            filename = tf_dir + '/' + split_name + '.tfrecord'
            print('Writing file {}...'.format(filename))
            with tf.python_io.TFRecordWriter(filename) as writer:
                for i in tqdm(range(dataset.num_samples)):
                    arrays, array_labels, img_names = sess.run([net, labels, image_names])
                    array, array_label, img_name = np.array(arrays[0]), array_labels[0], img_names[0].decode()
                    array_to_tfrecord(array, array_label, img_name, writer)
            print('{} complete.'.format(filename))
            
            
def tv_maker_avg(MODEL, split_name, dataset_dir, file_pattern, tf_dir, checkpoint_file, log_dir):
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)

        # load model
        model = nets_factory.get_network_fn(MODEL, num_classes=10, is_training=False)
        # get image size
        image_size = model.default_image_size


        dataset = get_split(split_name=split_name, 
                            dataset_dir=dataset_dir, 
                            file_pattern=file_pattern, 
                            file_pattern_for_counting='drivers', 
                            items_to_descriptions=None)
        images, _, labels, image_names = load_batch(
            dataset=dataset, 
            batch_size=1, 
            MODEL=MODEL, 
            height=image_size, 
            width=image_size, 
            is_training=False)

        _, endpoints = model(images)

        base_scope = base_layer(MODEL)

        net = endpoints[base_scope]
        shape = net.get_shape()[1:]
        net = tf.layers.average_pooling2d(inputs=net,
                                          pool_size=shape[:2],
                                          strides=1,
                                          padding='valid',
                                          name='Avg_pool')
        # net = tf.squeeze(net, [1, 2], name='Squeeze')
        shape = net.get_shape()[1:]
        write_shape_file(shape, MODEL, tf_dir)

        exclusion = exclude_scope(MODEL)
        variables_to_restore = slim.get_variables_to_restore(exclude=exclusion)
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        sv = tf.train.Supervisor(logdir=log_dir, summary_op=None, saver=None, init_fn=restore_fn)

        with sv.managed_session() as sess:
            filename = tf_dir + '/' + split_name + '.tfrecord'
            print('Writing file {}...'.format(filename))
            with tf.python_io.TFRecordWriter(filename) as writer:
                for i in tqdm(range(dataset.num_samples)):
                    arrays, array_labels, img_names = sess.run([net, labels, image_names])
                    array, array_label, img_name = np.array(arrays[0]), array_labels[0], img_names[0].decode()
                    array_to_tfrecord(array, array_label, img_name, writer)
            print('{} complete.'.format(filename))
    
####################################################
###            transfer-value loder              ###
####################################################   

def get_split_tv(split_name, dataset_dir, file_pattern=None, items_to_descriptions=None):
    '''
    Obtains the split - training or validation or test - to create a Dataset class for feeding the examples into a queue later on. 
    This function will set up the decoder and dataset information all into one Dataset class so that you can avoid the brute work later on.
    Your file_pattern is very important in locating the files later.
    INPUTS:
    - split_name(str): 'train' or 'validation'. Used to get the correct data split of tfrecord files
    - dataset_dir(str): the dataset directory where the tfrecord files are located
    - file_pattern(str): the file name structure of the tfrecord files in order to get the correct data
    - file_pattern_for_counting(str): the string name to identify your tfrecord files for counting
    OUTPUTS:
    - dataset (Dataset): A Dataset class object where we can read its various components for easier batch creation later.
    '''

    # First check whether the split_name is train or validation
    if split_name not in ['train', 'validation', 'test']:
        raise ValueError('split name {} was not recognized.'.format(split_name))

    #Create the full path for a general file_pattern to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern.format(split_name))

    #Count the total number of examples in all of these shard
    num_samples = 0
    file_pattern_for_counting = split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    if len(tfrecords_to_count) == 0:
        raise ValueError('There is no dataset.')
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    #Create a reader, which must be a TFRecord reader in this case
    reader = tf.TFRecordReader

    #Create the keys_to_features dictionary for the decoder
    keys_to_features = {
        'array/encoded': tf.FixedLenFeature([], tf.string),
        'array/filename': tf.FixedLenFeature([], tf.string),
        'array/class/label': tf.FixedLenFeature([], tf.int64)
        }
    items_to_handlers = {
        'array': slim.tfexample_decoder.Tensor('array/encoded'),
        'filename': slim.tfexample_decoder.Tensor('array/filename'),
        'label': slim.tfexample_decoder.Tensor('array/class/label')
        }

    #Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    
    #Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources = file_pattern_path,
        decoder = decoder,
        reader = reader,
        num_samples = num_samples,
        items_to_descriptions = items_to_descriptions)

    return dataset


def load_batch_tv(dataset, batch_size, MODEL, shape, is_training=True):
    '''
    Loads a batch for training.
    INPUTS:
    - dataset(Dataset): a Dataset class object that is created from the get_split function
    - batch_size(int): determines how big of a batch to train
    - height(int): the height of the image to resize to during preprocessing
    - width(int): the width of the image to resize to during preprocessing
    - is_training(bool): to determine whether to perform a training or evaluation preprocessing
    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).
    '''
    #First create the data_provider object
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        # common_queue_capacity = 24 + 3 * batch_size,
        common_queue_capacity = 2 * batch_size,
        common_queue_min = 1,
        shuffle=is_training)

    #Obtain the raw image using the get method
    array, filename, label = data_provider.get(['array', 'filename', 'label'])

    array = tf.decode_raw(array, tf.float32)
    array = tf.reshape(array, shape)

    #Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    arrays, labels, filenames = tf.train.batch(
        [array, label, filename],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size,
        allow_smaller_final_batch = True)

    return arrays, labels, filenames
    
    
    