import os
import tensorflow as tf
from tensorflow.contrib import slim
from preprocessing.preprocessing_factory import get_preprocessing
from nets import nets_factory
from datasets import dataset_utils

def get_split(split_name, dataset_dir, file_pattern=None, file_pattern_for_counting='drivers', items_to_descriptions=None):
    '''
    Obtains the split - training or validation - to create a Dataset class for feeding the examples into a queue later on. This function will
    set up the decoder and dataset information all into one Dataset class so that you can avoid the brute work later on.
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
    file_pattern_for_counting = file_pattern_for_counting + '_' + split_name
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
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string),
        'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'image/name': tf.FixedLenFeature((), tf.string)
    }

    #Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
        'image_name': slim.tfexample_decoder.Tensor('image/name'),
    }

    #Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    
    labels_to_names = None
    if split_name != 'test':
        #Create the labels_to_name file
        if dataset_utils.has_labels(dataset_dir):
            labels_to_names = dataset_utils.read_label_file(dataset_dir)

        num_classes = len(labels_to_names)
    else:
        num_classes = None

    #Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources = file_pattern_path,
        decoder = decoder,
        reader = reader,
        # num_readers = 4,
        num_samples = num_samples,
        num_classes = num_classes,
        labels_to_names = labels_to_names,
        items_to_descriptions = items_to_descriptions)

    return dataset

def load_batch(dataset, batch_size, MODEL, height, width, is_training=True):
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
    raw_image, label, image_name = data_provider.get(['image', 'label', 'image_name'])

    #Perform the correct preprocessing for this image depending if it is training or evaluating
    preprocessing_fn = get_preprocessing(MODEL, is_training)
    image = preprocessing_fn(raw_image, height, width)

    #As for the raw images, we just do a simple reshape to batch it up
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    #Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, raw_images, labels, image_names = tf.train.batch(
        [image, raw_image, label, image_name],
        batch_size=batch_size,
        # num_threads = 4,
        num_threads=1,
        # capacity = 4 * batch_size,
        capacity=2 * batch_size,
        allow_smaller_final_batch = True)

    return images, raw_images, labels, image_names