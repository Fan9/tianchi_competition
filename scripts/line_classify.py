#coding=utf-8
import tensorflow as tf
import random
import numpy as np
import time
from datetime import datetime


def read_data(data_dir):

    inf = []
    with open(data_dir,'r') as f:
        lines = f.readlines()
    for line in lines:

        line = [float(ele.strip()) for ele in line[1:-2].split(',')]
        inf.append(line)
    random.seed(123)
    random.shuffle(inf)
    inf = np.array(inf)
    num_train = int(len(inf)*0.8)
    train_data = inf[:num_train,:-1]
    train_label = inf[:num_train,-1]
    #train_label = [int(x) for x in train_label]
    test_data = inf[num_train:,:-1]
    test_label = inf[num_train:,-1]
    #test_label = [int(x) for x in test_label]
    #print(inf.shape,train_label.shape,train_data.shape,test_label.shape,test_data.shape)

    return train_data,train_label,test_data,test_label


#read_data('/home/fangsh/test/inf.txt')
def get_batch_data(data_dir,batch_size,step):
    train_data, train_label, test_data, test_label = read_data(data_dir)
    data_size = len(train_data)
    num_batches_per_epoch = int(data_size / batch_size) +1
    i = step%num_batches_per_epoch

    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, data_size)

    return train_data[start_index:end_index],train_label[start_index:end_index],test_data,test_label
    #return train_data, train_label, test_data, test_label


def model(x):


    with tf.variable_scope('fc1'):
        weights = tf.get_variable(initializer=tf.truncated_normal_initializer(mean=0.0,stddev=1.0,dtype=tf.float32),
                                  name='weight',shape=[25,20])
        bias = tf.get_variable(initializer=tf.constant_initializer(0.0),name='bias',shape=[20])
        net = tf.nn.relu(tf.matmul(x,weights)+bias)
        tf.summary.histogram('fc1/weights',weights)

    with tf.variable_scope('fc2'):
        weights = tf.get_variable(initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32),
                                  name='weight', shape=[20, 1])
        bias = tf.get_variable(initializer=tf.constant_initializer(0.0), name='bias', shape=[1])
        logits = tf.matmul(net, weights) + bias
        tf.summary.histogram('fc2/weights',weights)
        return logits
    '''
    global_step = tf.train.get_or_create_global_step()
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y,logits=logits)
    tf.summary.scalar('total_loss',loss)
    lr = tf.train.exponential_decay(0.1,global_step,10000,0.8)
    tf.summary.scalar('learn_rate',lr)
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    '''
    #print(logits)
    #pre = tf.equal(tf.arg_max(logits,1),y)
    #accuracy = tf.reduce_mean(tf.cast(pre,'float'))

def loss(y,logits):

    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y,logits=logits)
    tf.summary.scalar('total_loss',loss)
    return  loss

def train_op(loss):
    global_step = tf.train.get_or_create_global_step()
    lr = tf.train.exponential_decay(0.1,global_step,10000,0.8)
    tf.summary.scalar('learn_rate',lr)
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    return train_op


def train(data_dir):

    tf.reset_default_graph()
    with tf.variable_scope('input'):
        x = tf.placeholder(dtype=tf.float32,shape=[None,25],name='input')
        y = tf.placeholder(dtype=tf.int64,shape=[None,1],name='label')

    logits = model(x)
    losses = loss(y,logits)
    train_ = train_op(losses)

    saver = tf.train.Saver(max_to_keep=2)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        writer = tf.summary.FileWriter(logdir='/home/fangsh/test/log/', graph=sess.graph)
        merged = tf.summary.merge_all()

        start_time = time.time()
        for i in range(50000):


            train_data,train_label,test_data,test_label = get_batch_data(data_dir,8,i)
            train_label = train_label[:,np.newaxis]
            test_label = test_label[:, np.newaxis]
            _,l,summary= sess.run([train_,losses,merged],feed_dict={x:train_data,y:train_label})
            saver.save(sess=sess,save_path='/home/fangsh/test/log/my_model.ckpt',global_step=i)
            #传入global_step后默认最多保存五个checkpoint
            writer.add_summary(summary=summary,global_step=i)


            if i%100 == 0:
                current_time = time.time()
                duration = current_time - start_time

                format_str = '%s:    step:%d%s        loss:%03.5f   (%.3fs/batch)'
                print(format_str%(datetime.now(),i,' '*(5-len(str(i))),l,duration/100))
                pre = sess.run([logits],feed_dict={x:test_data,y:test_label})
                pre = np.squeeze(pre)
                pre = [1 if ele > 0.5 else 0 for ele in pre]

                true_num = 0
                for idx in range(len(pre)):
                    if pre[idx] == test_label[idx]:
                        true_num += 1
                acc = true_num/len(pre)
                start_time = current_time
                print('%s:    step:%d%s        acc:%03.5f    (%.3fs/batch)' % (datetime.now(),i,' '*(5-len(str(i))),acc,duration/100))


if __name__ == '__main__':
    data_dir = '/home/fangsh/test/inf.txt'
    train(data_dir)
