from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

dataset_name = 'tianchi'
dataset_split_name = 'train'
dataset_dir = '/home/fangsh/tianchi/tianchi_dataset/tfrecord'
batcg_size = 32


dataset = dataset_factory.get_dataset(
        dataset_name, dataset_split_name, dataset_dir)

provider = slim.dataset_data_provider.DatasetDataProvider(
    dataset,
    num_readers= 4,
    common_queue_capacity=20 * batch_size,
    common_queue_min=10 * batch_size)
[image, label] = provider.get(['image_raw', 'label'])