# coding=utf-8
import tensorflow as tf
import os
import numpy as np
import sys
import xml.etree.ElementTree as ET
import random


tf.app.flags.DEFINE_string('root_dir',
                           '/home/fangsh/tianchi/tianchi_dataset/data_deal/data',
                           'Path to root')

tf.app.flags.DEFINE_string('outp_dir',
                           '/home/fangsh/tianchi/tianchi_dataset/data_deal/data/tfrecord',
                           'Path to save tfrecord file')

tf.app.flags.DEFINE_string('cls_dir',
                           '/home/fangsh/tianchi/tianchi_dataset/tfrecord/class.txt',
                           'Path to save classes file')

FLAGS = tf.app.flags.FLAGS


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _convert_to_example(image_raw, lab):

    image_format = b'PNG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_raw),
        'image/format': bytes_feature(image_format),
        'label': int64_feature(lab)
    }))
    return example


def main(args=None):
    data_split = ['train', 'test']
    data_cls = ['norm', 'xiaci']
    # for i, c in enumerate(data_cls):

    # split norm data
    norm_data = os.listdir(os.path.join(FLAGS.root_dir, 'norm'))
    #norm_data = np.array(norm_data)
    random.seed(123)
    random.shuffle(norm_data)
    train_norm_data = norm_data[:3040]
    test_norm_data = norm_data[3040:3800]

    # split xiaci
    xiaci_data = os.listdir(os.path.join(FLAGS.root_dir, 'xiaci'))
    #xiaci_data = np.array(xiaci_data)
    random.seed(123)
    random.shuffle(xiaci_data)
    train_xiaci_data = xiaci_data[0:1550]
    test_xiaci_data = xiaci_data[1550:1937]

    for i, c in enumerate(data_split):

        sys.stdout.write('\n cls:%s, lab:%d' % (c, i))
        cur_outp_dir = os.path.join(FLAGS.outp_dir, '%s.tfrecord' % c)
        with tf.python_io.TFRecordWriter(cur_outp_dir) as writer:
            if c == 'train':
                cur_norm_data = train_norm_data
                cur_xiaci_data = train_xiaci_data
            else:
                cur_norm_data = test_norm_data
                cur_xiaci_data = test_xiaci_data
            #print(cur_norm_data, cur_xiaci_data)
            for idx, image in enumerate(cur_norm_data):
                #print(image)
                #print(norm_data)
                sys.stdout.write('\r >> Convert Norm Image  %d/%d' % (idx + 1, len(cur_norm_data)))
                sys.stdout.flush()
                cur_image_dir = os.path.join(FLAGS.root_dir, 'norm', image)
                image_raw = tf.gfile.FastGFile(cur_image_dir, 'rb').read()
                example = _convert_to_example(image_raw, lab=0)
                writer.write(example.SerializeToString())

            sys.stdout.write('\n')
            for idx, image in enumerate(cur_xiaci_data):

                sys.stdout.write('\r >> Convert Xiaci Image  %d/%d' % (idx + 1, len(cur_xiaci_data)))
                sys.stdout.flush()

                cur_image_dir = os.path.join(FLAGS.root_dir, 'xiaci', image)
                image_raw = tf.gfile.FastGFile(cur_image_dir, 'rb').read()
                example = _convert_to_example(image_raw, lab=1)
                writer.write(example.SerializeToString())

    print('\nFinished convert tianchi dataset!')


if __name__ == '__main__':
    tf.app.run()
