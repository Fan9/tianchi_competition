from line_classify import model
import tensorflow as tf
import numpy as np

def read_data(data_dir):

    data = []
    with open(data_dir,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = [float(ele.strip()) for ele in line[1:-2].split(',')]
            data.append(line)
    return data


def eval(checkpoint_path,data_dir):

    x = tf.placeholder(dtype=tf.float32,shape=[None,25],name='input')
    pred = model(x)
    probability = tf.sigmoid(pred)

    saver = tf.train.Saver()
    with tf.Session() as sess:

        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
        x2 = read_data(data_dir)
        print(x2)
        prediction = sess.run([probability],feed_dict={x:x2})
        prediction = np.squeeze(prediction)
        with open('/home/fangsh/test/sub.txt','a') as f:
            for ele in prediction:
                f.write(str(ele)+'\n')
        print(prediction)


if __name__ == '__main__':
    data_dir = '/home/fangsh/test/data.txt'
    checkpoint_path = '/home/fangsh/test/log-old'
    eval(checkpoint_path,data_dir)