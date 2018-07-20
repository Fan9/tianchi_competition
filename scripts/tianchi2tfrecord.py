#coding=utf-8
import os
import glob
import tensorflow as tf
import sys
import xml.etree.ElementTree as ET


tf.app.flags.DEFINE_string('root_dir',
                           '/home/fangsh/tianchi/tianchi_dataset/data_megred/train',
                           'Path to root')

tf.app.flags.DEFINE_string('file_dir',
                           '/home/fangsh/tianchi/tianchi_dataset/tfrecord/train.tfrecord',
                           'Path to save tfrecord file')

tf.app.flags.DEFINE_string('cls_dir',
                           '/home/fangsh/tianchi/tianchi_dataset/classes.txt',
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


def _process_per_image(image_dir,xml_dir,lab):

    image_raw = tf.gfile.FastGFile(image_dir,'rb').read()
    tree = ET.parse(xml_dir)
    root = tree.getroot()
    size = root.find('size')
    shape=[int(size.find('height').text),
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
        bboxes.append((float(bbox[0].text)/shape[1],
                       float(bbox[1].text)/shape[0],
                       float(bbox[2].text)/shape[1],
                       float(bbox[3].text)/shape[0]))
        #print(bboxes)
    return (image_raw,lab,shape,object_name,truncated,difficult,bboxes)


def _convert_to_example(image_raw,lab,shape,object_name,truncated,
                        difficult,bboxes):
    bboxes = list(map(list,zip(*bboxes)))
    #print(bboxes)
    iter_bboxes = iter(bboxes)
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/shape': int64_feature(shape),
        'label':int64_feature(lab),
        'image_raw':bytes_feature(image_raw),
        'object/name': bytes_feature(object_name),
        'object/truncated':int64_feature(truncated),
        'object/difficult':int64_feature(difficult),
        'object/xmin':float_feature(next(iter_bboxes,[])),
        'object/ymin':float_feature(next(iter_bboxes,[])),
        'object/xmax':float_feature(next(iter_bboxes,[])),
        'object/ymax':float_feature(next(iter_bboxes,[]))
         }))
    return example

def main(args=None):

    #get all classes
    root_dir = FLAGS.root_dir
    file_dir = FLAGS.file_dir
    classes = os.listdir(root_dir)
    with tf.python_io.TFRecordWriter(file_dir) as writer:
        for lab,cls in enumerate(classes):

            sys.stdout.write('\n')
            #dict to save classes info
            class_dict={}
            class_dict[lab] = cls
            with open(FLAGS.cls_dir,'a') as f:
                for key,value in class_dict.items():
                    f.write(str(key)+': '+value+'\n')


            cur_cls_dir = os.path.join(root_dir,cls,'*.jpg')
            images_dir = glob.glob(cur_cls_dir)


            for idx,image_dir in enumerate(images_dir):


                #sys.stdout.write('%s\n'%(images_dir))
                format_str = '\r  >> converting the cls: %s%s%d/%d'
                sys.stdout.write(format_str % (cls, (10-len(cls))*' ', idx+1, len(images_dir)))
                sys.stdout.flush()
                if cls == '正常':
                    image_raw = tf.gfile.FastGFile(image_dir,'rb').read()
                    l_data= (image_raw,lab,[1920,2560,3],['none'.encode('utf-8')],\
                             [-1],[-1],[(-1.0,-1.0,-1.0,-1.0)])
                    example = _convert_to_example(*l_data)
                    writer.write(example.SerializeToString())

                else:
                    name = image_dir.split('/')[-1][0:-4]
                    xml_dir = os.path.join(root_dir,cls,name+'.xml')
                    l_data = _process_per_image(image_dir,xml_dir,lab)
                    example = _convert_to_example(*l_data)
                    writer.write(example.SerializeToString())
    print('\nFinished converting the Tianchi dataset!')


if __name__ == '__main__':
    tf.app.run()