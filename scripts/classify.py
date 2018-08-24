import tensorflow as tf
import h5py
import numpy as np
slim = tf.contrib.slim
import os
#import random
#from model1 import model

def read_data():
    # resnet_50_file_dir = '/media/fangsh/data/tianchi/feature_map/resnet_inception/resnet_50.h5'
    # inception_v3_file_dir = '/media/fangsh/data/tianchi/feature_map/resnet_inception/inception_v3.h5'
    # with h5py.File(resnet_50_file_dir,'r') as h:
    #     data1 = np.array(h['train'])
    #     label1 = np.array(h['label'])
    # with h5py.File(inception_v3_file_dir,'r') as h:
    #     data2 = np.array(h['train'])
    #     label2 = np.array(h['label'])
    #
    # data = np.zeros([7381,4096],dtype=np.float32)
    # data[:,:2048] = data1
    # data[:,2048:4096] = data2

    #for idx in range(len(label)):
        #if label[idx]==1:
    file_dir = '/media/fangsh/data/tianchi/feature_map/merged_train_8_3.h5'
    with h5py.File(file_dir) as h5:
        data = np.array(h5['train'])
        label = np.array(h5['label'])


    np.random.seed(2018)
    np.random.shuffle(data)
    np.random.seed(2018)
    np.random.shuffle(label)

    #np.random.seed(2018)
    #np.random.shuffle(data)
    #np.random.seed(2018)
    #np.random.shuffle(label)

    return data[:1618],label[:1618],data[1618:],label[1618:]

def get_batch_data(train_data,label,batch_size,step):
    nums_batch_per_epoch = int(len(train_data)/batch_size)
    i = step%nums_batch_per_epoch
    start_index = i*batch_size
    end_index = min((i+1)*batch_size,len(train_data))

    return train_data[start_index:end_index],label[start_index:end_index]


def model(inputs,is_training=True,dropout_keep_prob=0.5,spatial_squeeze=True):

    with tf.variable_scope('my_model',[inputs]) as sc:

        end_points_collection = 'model_endpoints'
        with slim.arg_scope([slim.fully_connected,slim.conv2d],
                            outputs_collections=end_points_collection,
                            activation_fn=tf.nn.relu,
                            #weights_initializer=tf.truncated_normal_initializer(mean=0,stddev=1.0),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=tf.zeros_initializer()
                            ):
            with slim.arg_scope([slim.conv2d],padding='SAME'):

                #net = slim.dropout(inputs,dropout_keep_prob,is_training=is_training,scope='dropout')
                #net = slim.batch_norm()
                net = slim.repeat(inputs,2,slim.conv2d,768,[3,3],scope='conv1')
                net = slim.max_pool2d(net,[2,2],stride=2, padding='VALID',scope='pool1')
                net = slim.conv2d(net,768,[3,3],scope='conv2')
                net = slim.max_pool2d(net,[2,2],stride=2, padding='VALID',scope='pool2')
                net = slim.conv2d(net,768,[3,3],scope='conv3')
                net = slim.conv2d(net,1024,[8,8],padding='VALID',scope='fc2')
                net = slim.dropout(net, 0.5, is_training=is_training,
                                   scope='dropout1')
                net = slim.conv2d(net,1024,[1,1],scope='fc3')
                net = slim.dropout(net, 0.5, is_training=is_training,
                                   scope='dropout2')
                #net = slim.avg_pool2d(net,[8,8],scope='gap_pool')
                #net = slim.dropout
                net = slim.conv2d(net,2,[1,1],activation_fn=None,
                                  normalizer_fn=None,
                                  scope='fc1')
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if spatial_squeeze:
                    net = tf.squeeze(net,[1,2],name='fc/squeeze')
                    end_points[sc.name+'fc1'] = net
    return net,end_points


def train(h5file_dir):

    x = tf.placeholder(shape=[None,35,35,768], dtype=tf.float32,name='input')
    y = tf.placeholder(shape=[None], dtype=tf.int64,name='label')
    is_training = tf.placeholder(dtype=tf.bool)
    y_,end_p = model(x,is_training)
    y_ = tf.squeeze(y_)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_)
    tf.summary.scalar('vgg/losses', loss)
    top_k_op = tf.nn.in_top_k(y_,y,1,name='top_1_op')
    global_step = tf.train.get_or_create_global_step()
    lr = tf.train.exponential_decay(learning_rate=1e-3,
                                    global_step=global_step,
                                    decay_steps=1000,
                                    decay_rate=0.9)
    tf.summary.scalar('vgg/learning_rate',lr)
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        train_data, train_label, test_data, test_label = read_data()

        writer = tf.summary.FileWriter('/home/fangsh/test/train_dir', sess.graph)
        merged_summary = tf.summary.merge_all()


        for step in range(50000):

            train_batch,label_batch = get_batch_data(train_data,train_label,32,step)

            _,losses,summary = sess.run([train_op,loss,merged_summary],feed_dict={x:train_batch,y:label_batch,is_training:True})
            writer.add_summary(summary,step)


            if step%10 == 0:
                print('step:%d,   loss:%.6f'%(step,losses))
                logits,pred = sess.run([y_,top_k_op],feed_dict={x:test_data,y:test_label,is_training:False})

                true_num = 0
                true_num +=np.sum(pred)
                acc = true_num / len(pred)
                print('--------------------------')
                #print(logits)
                print('--------------------------')
                print('----------------->> acc:%.6f'%acc)
            if step%50 == 0:
                checkpoint_save_path = '/home/fangsh/test/train_dir/my_model.ckpt'
                saver.save(sess, checkpoint_save_path, global_step=step)



if __name__ == '__main__':
    h5file_dir = '/home/fangsh/test/merged.h5'
    train(h5file_dir)
