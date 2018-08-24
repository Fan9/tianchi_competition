from classify import model
import tensorflow as tf
import numpy as np
import h5py
np.set_printoptions(suppress=True)
from tqdm import tqdm



def get_test_data(h5file_dir):
    with h5py.File(h5file_dir) as h5:
        test_data = np.array(h5['test'])

    return test_data


def get_batch_test_data(test_data,batch_size,step):
    nums_batch_per_epoch = int(len(test_data) / batch_size) + 1
    i = step % nums_batch_per_epoch
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, len(test_data))

    return test_data[start_index:end_index]

def eval(h5file_dir,checkpoint_path):
    x = tf.placeholder(dtype=tf.float32,shape=[None,35,35,768],name='input')
    pred,endpoints = model(x,is_training=False)
    pro = tf.nn.softmax(pred)


    with tf.Session() as sess:



        saver = tf.train.Saver()
        saver.restore(sess,checkpoint_path)
        batch_size = 32
        test = get_test_data(h5file_dir=h5file_dir)
        num_batch = np.ceil(len(test)/batch_size)
        for step in tqdm(range(int(num_batch))):
            test_data_batch = get_batch_test_data(test,batch_size,step)
            pre = sess.run(pro,feed_dict={x:test_data_batch})
            print(pre.shape)
           # print(pre)
            result = ['{:.6f}'.format(ele) for ele in pre[:,1]]
            #print(result)
            for idx,ele in enumerate(result):
                if float(ele) > 0.999998:
                    result[idx] = 0.999981
                if float(ele) < 0.000001 :
                    result[idx] = '0.000031'


            with open('/home/fangsh/test/result0731.txt','a') as f:

                 for ele in result:
                     f.write(str(ele)+'\n')

            #print(pre.shape)
            #print(pre)


if __name__ == '__main__':
    h5file_dir = '/home/fangsh/test/merged_test.h5'
    checkpoint_path = '/home/fangsh/test/log3/my_model.ckpt-2500'
    eval(h5file_dir,checkpoint_path)
