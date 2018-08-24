import tensorflow as tf


slim = tf.contrib.slim
def model(inputs,is_training):
    with tf.variable_scope('my_model', [inputs]) as sc:
        with slim.arg_scope([slim.fully_connected, slim.conv2d],
                            #outputs_collections=end_points_collection,
                            activation_fn=tf.nn.relu,
                            # weights_initializer=tf.truncated_normal_initializer(mean=0,stddev=1.0),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=tf.zeros_initializer()
                            ):

            net = slim.dropout(inputs,keep_prob=0.5,is_training=is_training,scope='dropout')
            net = slim.fully_connected(net,2,activation_fn=None,normalizer_fn=None)
    return net


    # with tf.variable_scope('fc1'):
    #     weights = tf.get_variable(initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32),
    #                               name='weight', shape=[4096, 2])
    #     bias = tf.get_variable(initializer=tf.constant_initializer(0.0), name='bias', shape=[2])
    #     net = tf.nn.relu(tf.matmul(x, weights) + bias)
    #     tf.summary.histogram('fc1/weights', weights)
    #
    # with tf.variable_scope('fc2'):
    #     weights = tf.get_variable(initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32),
    #                               name='weight', shape=[20, 1])
    #     bias = tf.get_variable(initializer=tf.constant_initializer(0.0), name='bias', shape=[1])
    #     logits = tf.matmul(net, weights) + bias
    #     tf.summary.histogram('fc2/weights', weights)
    #
    # return logits