#coding=utf-8
import tensorflow as tf
import os
import numpy as np
import sys
import xml.etree.ElementTree as ET


tf.app.flags.DEFINE_string('root_dir',
                           '/home/fangsh/tianchi/tianchi_dataset/data_megred/train',
                           'Path to root')

tf.app.flags.DEFINE_string('outp_dir',
                           '/home/fangsh/tianchi/tianchi_dataset/tfrecord',
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


def _process_per_image(image_dir, xml_dir, lab):
    image_raw = tf.gfile.FastGFile(image_dir, 'rb').read()
    tree = ET.parse(xml_dir)
    root = tree.getroot()
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]

    object_name = []
    truncated = []
    difficult = []
    bboxes = []

    for obj in root.findall('object'):
        object_name.append(obj.find('name').text.encode('utf-8'))
        truncated.append(int(obj.find('truncated').text))
        difficult.append(int(obj.find('difficult').text))
        bbox = obj.find('bndbox')
        bboxes.append((float(bbox[0].text) / shape[1],
                       float(bbox[1].text) / shape[0],
                       float(bbox[2].text) / shape[1],
                       float(bbox[3].text) / shape[0]))
        # print(bboxes)
    return (image_raw, lab, shape, object_name, truncated, difficult, bboxes)


def _convert_to_example(image_raw, lab, shape, object_name, truncated,
                        difficult, bboxes):
    bboxes = list(map(list, zip(*bboxes)))
    # print(bboxes)
    iter_bboxes = iter(bboxes)
    image_format = b'PNG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded':bytes_feature(image_raw),
        'image/format':bytes_feature(image_format),
        'image/shape': int64_feature(shape),
        'label': int64_feature(lab),
        'object/name': bytes_feature(object_name),
        'object/truncated': int64_feature(truncated),
        'object/difficult': int64_feature(difficult),
        'object/bbox/xmin': float_feature(next(iter_bboxes, [])),
        'object/bbox/ymin': float_feature(next(iter_bboxes, [])),
        'object/bbox/xmax': float_feature(next(iter_bboxes, [])),
        'object/bbox/ymax': float_feature(next(iter_bboxes, []))
    }))
    return example


def main(args=None):
    cls_info = []
    file_dir = []
    data_split = ['train.tfrecord', 'test.tfrecord']

    for root, sub, files in os.walk(FLAGS.root_dir):
        cls_info.extend(sub)
        for file in files:
            file_dir.append(os.path.join(root, file))
    #print(cls_info)

    #write class info to class.txt
    with open(FLAGS.cls_dir, 'a') as f:
        for idx, cls in enumerate(cls_info):
            f.write(str(idx) + ':' + cls + '\n')

    image_dir = []
    for i in file_dir:
        #print(i.split('/')[-1][-4:])
        if i.split('/')[-1][-4:] == '.jpg':
            image_dir.append(i)

    image_dir = np.array(image_dir)
    np.random.seed(123)
    np.random.shuffle(image_dir)

    num_train_image = int(0.7 * len(image_dir))

    train_image_dir = image_dir[0:num_train_image]
    var_image_dir = image_dir[num_train_image:]

    for i in data_split:
        cur_out_dir = os.path.join(FLAGS.outp_dir, i)
        sys.stdout.write('\n')
        with tf.python_io.TFRecordWriter(cur_out_dir) as writer:
            if i.split('.')[0] == 'train':
                data_dir = train_image_dir
            else:
                data_dir = var_image_dir


            num_negivate = 0
            num_positive = 0
            for idx, cur_image_dir in enumerate(data_dir):

                sys.stdout.write('\r >> Convert to %s   %d/%d'%(i, idx + 1, len(data_dir)))
                cur_cls = cur_image_dir.split('/')[-2]
                #lab = cls_info.index(cur_cls)
                if cur_cls == '正常':
                    lab = 0
                    num_negivate +=1
                    image_raw = tf.gfile.FastGFile(cur_image_dir, 'rb').read()
                    l_data = (image_raw, lab, [1920, 2560, 3], ['none'.encode('utf-8')], \
                          [-1], [-1], [(0.0, 0.0, 1.0, 1.0)])
                    example = _convert_to_example(*l_data)
                    writer.write(example.SerializeToString())
                else:

                    lab = 1
                    num_positive +=1
                    xml_dir = cur_image_dir[0:-4] + '.xml'

                    l_data = _process_per_image(cur_image_dir, xml_dir, lab)
                    example = _convert_to_example(*l_data)
                    writer.write(example.SerializeToString())

            sys.stdout.write('\n       positive num:%d'%num_positive)
            sys.stdout.write('\n       negivate num:%d'%num_negivate)
            sys.stdout.write('\n       pos/neg ratio:%.2f'%(num_positive/num_negivate))
    print('\nFinished convert tianchi dataset!')

if __name__ == '__main__' :
    tf.app.run()