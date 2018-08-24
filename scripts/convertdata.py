import os
import random
import tensorflow as tf
import sys
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt


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

    image_format = b'jpg'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_raw),
        'image/format': bytes_feature(image_format),
        'label': int64_feature(lab)
    }))
    return example

def get_image_dirs(root_dir):

    file_dir = []
    norm_images_dir = []
    xiaci_images_dir = []
    for root, sub, files in os.walk(root_dir):
        for file in files:
            file_dir.append(os.path.join(root, file))

    for f in file_dir:
        if f[-4:] == '.jpg':

            if f.split('/')[-2] == '正常':
                norm_images_dir.append(f)
            else:
                xiaci_images_dir.append(f)

    return norm_images_dir,xiaci_images_dir



def main():

    outp_dir = '/home/fangsh/tianchi/tianchi_dataset/tfrecord/train.tfrecord'
    #data_dir = '/home/fangsh/tianchi/tianchi_dataset/data_deal_3x3/selected_crop_xiaci/merged_norm_xiaci'
    root_dir = '/home/fangsh/tianchi/tianchi_dataset/data_megred/train'
    #data_split = ['train.tfrecord', 'test.tfrecord']


    norm_images_dir,xiaci_images_dir = get_image_dirs(root_dir)

    random.seed(123)
    random.shuffle(norm_images_dir)
    random.seed(123)
    random.shuffle(xiaci_images_dir)


    with tf.python_io.TFRecordWriter(outp_dir) as writer:


        for img_dir in tqdm(norm_images_dir):
            #clu_cls = img_dir.split('/')[-2]

            label = 0
            img_raw = tf.gfile.FastGFile(img_dir, 'rb').read()
            image = tf.image.decode_jpeg(img_raw)
            image = data_aug(image)
            img_raw1 = tf.image.encode_jpeg(image)
           # image1 = tf.image.decode_jpeg(img_raw1)

            example = _convert_to_example(img_raw1, label)

            for i in range(3):
                writer.write(example.SerializeToString())

        for img_dir in tqdm(xiaci_images_dir):
            label = 1
            img_raw = tf.gfile.FastGFile(img_dir, 'rb').read()
            example = _convert_to_example(img_raw, label)

            for i in range(6):
                writer.write(example.SerializeToString())


    print('\nFinished convert tianchi dataset!')


if __name__ == '__main__':
    main()
