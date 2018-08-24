#coding=utf-8
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2

file_dir = '/home/fangsh/tianchi/tianchi_dataset/tfrecord/train.tfrecord'

file_queue = tf.train.string_input_producer([file_dir])
reader = tf.TFRecordReader()
_,serialized = reader.read(file_queue)
feature = tf.parse_single_example(serialized=serialized,
                                  features={
                                      'image/encoded':tf.FixedLenFeature([],tf.string),
                                      'image/format':tf.FixedLenFeature([],tf.string),
                                      'label': tf.FixedLenFeature([],tf.int64),
                                        })
label =  feature['label']
image_raw = feature['image/encoded']

#change 42 classes to 2 classes
#label = 0 if label==33 else 1

image = tf.image.decode_jpeg(image_raw)
label1 = tf.cast(label,tf.int32)
#float_image = tf.cast(image,tf.uint8)
#float_image =tf.reshape(float_image,[384,512,3])


#float_image1 = tf.reshape(image,[1920,2560,3])

#float_image.set_shape([1920,2560,3])
#  cant use :ValueError: Shapes (?,) and (1920, 2560, 3) are not compatible


#float_image.set_shape(shape)
# cant use :shape must be int,not tensor

#resized_image = tf.image.resize_images(float_image, [224, 224])
#resized_image1 = tf.cast(resized_image,tf.uint8)
# cant use :image must have 3/4 dimensions

with tf.Session() as sess:
    sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess,coord)
    for i in range(100):
        #_,img = sess.run([shape])
        #print(lab)
        #print(shape.eval()[0])
        #print(shape.eval()[1])
        #print(shape.eval()[2])
        #for i in name.values:
            #print(i.decode('utf-8'))
        print(label1.eval())
        #cv2.imshow('img',image.eval())
        #cv2.waitKey()
        plt.imshow(image.eval())
        #plt.imshow 格式要求  
        #3-dimensional arrays must be of dtype unsigned byte, unsigned short, float32 or float64
        plt.show()
    coord.request_stop()
    coord.join(threads)
