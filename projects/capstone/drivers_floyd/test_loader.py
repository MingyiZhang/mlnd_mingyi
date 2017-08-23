import os
from skimage.io import imread, imshow
import tensorflow as tf
from preprocessing.preprocessing_factory import get_preprocessing
from nets import nets_factory

def test_preprocessing(img_path, MODEL, height, width):
    '''
    preprocessing the testset.

    Args:
        img_path(str): path of image (contain image name)
        MODEL(str): pretrained model name.
        height(int): resize height.
        width(int): resize width

    Returns:
        img(Tensor): preprocessed image [1, height, width, channels]
        img_raw(narray): original image
    '''

    preprocessing_fn = get_preprocessing(MODEL, is_training=False)
    img_raw = imread(img_path)
    img = preprocessing_fn(img_raw, height, width)
    img = tf.expand_dims(img, 0)
    return img, img_raw

def test_generator(img_dir, MODEL, height, width):
    '''
    load the test image

    Args:
        img_dir(str): directory of testset
        MODEL(str): pretrained model name.
        height(int): resize height.
        width(int): resize width

    Returns:
        img_pp(Tensor): preprocessed image [1, height, width, channels]
        img_raw(narray): original image
    '''
    for img in os.listdir(img_dir):
        img_path = img_dir + '/' + img
        img_pp, img_raw = test_preprocessing(img_path, MODEL, height, width)
        yield img_pp, img_raw
