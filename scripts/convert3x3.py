import numpy as np
import tensorflow as tf
from tqdm import tqdm
#from convert_common import int64_feature,bytes_feature,float_feature
from convert_common import _convert_to_example
import os
import random

def convert():
    root_dir = '/home/fangsh/tianchi/tianchi_dataset/data_deal_3x3/selected_crop_xiaci'
    outp_dir = '/media/fangsh/data/tianchi/data/3x3tfrecord'


    #deal norm data
    norm_data_dir = os.path.join(root_dir,'norm')
    norm_data = os.listdir(norm_data_dir)
    norm_label = np.zeros(shape=[len(norm_data)],dtype=np.int16)

    #deal xiaci data
    xiaci_data_dir = os.path.join(root_dir,'xiaci_random_crop','xiaci')
    xiaci_data = os.listdir(xiaci_data_dir)
    xiaci_label = np.ones([len(xiaci_data)],dtype=np.int16)

    data = np.concatenate((np.array(norm_data),np.array(xiaci_data)))
    label = np.concatenate((norm_label,xiaci_label))

    random.seed(2018)
    random.shuffle(data)
    random.seed(2018)
    random.shuffle(label)

    train_radio = 0.8
    num = int(len(data)*0.8)

    # convert train tfrecord format
    train_outp_dir = os.path.join(outp_dir,'train.tfrecord')
    with tf.python_io.TFRecordWriter(train_outp_dir) as writer:
        for idx,image in enumerate(tqdm(data[0:num])):



            if label[idx] == 0:
                cur_image_dir = os.path.join(root_dir,'norm',image)
            else:
                cur_image_dir = os.path.join(root_dir,'xiaci_random_crop','xiaci',image)
            img_raw = tf.gfile.FastGFile(cur_image_dir,'rb').read()
            example = _convert_to_example(img_raw,label[idx])
            writer.write(example.SerializeToString())


    # convert xiaci tfrecord format
    train_outp_dir = os.path.join(outp_dir, 'test.tfrecord')
    with tf.python_io.TFRecordWriter(train_outp_dir) as writer:
        for idx, image in enumerate(tqdm(data[num:])):
            if label[idx+num] == 0:
                cur_image_dir = os.path.join(root_dir, 'norm', image)
            else:
                cur_image_dir = os.path.join(root_dir, 'xiaci_random_crop','xiaci', image)
            img_raw = tf.gfile.FastGFile(cur_image_dir, 'rb').read()
            example = _convert_to_example(img_raw, label[idx+num])
            writer.write(example.SerializeToString())

    print(' Finished convert tianchi dataset!')

if __name__ == '__main__':
    convert()