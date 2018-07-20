from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt

slim = tf.contrib.slim

dataset_name = 'tianchi'
dataset_split_name = 'train'
dataset_dir = '/home/fangsh/tianchi/tianchi_dataset/tfrecord'



dataset = dataset_factory.get_dataset(dataset_name,dataset_split_name,dataset_dir)

provider = slim.dataset_data_provider.DatasetDataProvider(
    dataset,
    num_readers=4,
    common_queue_capacity= 20*32,
    common_queue_min=10*32
)
[image,label,bbox] = provider.get(['image','label','object/bbox'])
img_data = tf.cast(image,tf.float32)
label = tf.cast(label,tf.int32)
bbox = tf.cast(bbox,tf.float32)

with tf.Session() as sess:
    #tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(10):
        lab,bx=sess.run([label,bbox])
        print(lab)
        print(bx)

        #print(bbox.eval())
        #print(label.eval())
        #plt.imshow(img_data.eval())
        #plt.show()
    coord.request_stop()
    coord.join(threads)
