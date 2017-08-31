import tensorflow as tf
from datasets import dataset_utils


def ckpt_maker(model, checkpoints_dir='checkpoints'):
    url_pattern = 'http://download.tensorflow.org/models/{}.tar.gz'

    file_pattern = checkpoints_dir + '/{}.ckpt'

    nets_tar_map = {'vgg_16': 'vgg_16_2016_08_28',
                    'vgg_19': 'vgg_19_2016_08_28',
                    'inception_v1': 'inception_v1_2016_08_28',
                    'inception_v2': 'inception_v2_2016_08_28',
                    'inception_v3': 'inception_v3_2016_08_28',
                    'inception_v4': 'inception_v4_2016_09_09',
                    'inception_resnet_v2': 'inception_resnet_v2_2016_08_30',
                    'resnet_v2_50': 'resnet_v2_50_2017_04_14',
                    'resnet_v2_101': 'resnet_v2_101_2017_04_14',
                    'resnet_v2_152': 'resnet_v2_152_2017_04_14',
                    }

    nets_ckpt_map = {'vgg_16': 'vgg_16',
                     'vgg_19': 'vgg_19',
                     'inception_v1': 'inception_v1',
                     'inception_v2': 'inception_v2',
                     'inception_v3': 'inception_v3',
                     'inception_v4': 'inception_v4',
                     'inception_resnet_v2': 'inception_resnet_v2_2016_08_30',
                     'resnet_v2_50': 'resnet_v2_50',
                     'resnet_v2_101': 'resnet_v2_101',
                     'resnet_v2_152': 'resnet_v2_152',
                    }

    if model not in nets_ckpt_map:
        raise ValueError('No checkpoints for {}.'.format(model))
    else:
        net_ckpt = nets_ckpt_map[model]
        net_tar = nets_tar_map[model]
        url = url_pattern.format(net_tar)
        checkpoint_file = file_pattern.format(net_ckpt)

    if not tf.gfile.Exists(checkpoints_dir):
        tf.gfile.MakeDirs(checkpoints_dir)

    if not tf.gfile.Exists(checkpoint_file):
        dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

    print('Checkpoint for {} is ready!'.format(model))
    print('File name: {}'.format(checkpoint_file))

    return checkpoint_file
