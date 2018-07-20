from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from datetime import datetime
import time
import os

import tensorflow as tf

import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', '/home/fangsh/tianchi/log',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    #with tf.device('/cpu:0'):
    images, labels = model.inputs(is_train=True)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = model.inference(images)

    # Calculate loss.
    loss = model.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.train(loss, global_step)
    #accuracy = model.accuracy(logits,labels)
    saver = tf.train.Saver()

    with tf.Session() as sess:
      tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()).run()
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess,coord=coord)
      writer = tf.summary.FileWriter(FLAGS.log_dir,sess.graph)
      merged_summary = tf.summary.merge_all()

      for step in range(FLAGS.max_steps):
        
        _,total_loss,summary = sess.run([train_op,loss,merged_summary])

        writer.add_summary(summary,global_step=step)
        if step % 20 == 0:
          #prediction = sess.run([accuracy],feed_dict={data_cls:'var1'})
          #acc = np.sum(prediction)/32
          format = '%s:  step %d, loss %.2f'
          print(format%(datetime.now(),step,total_loss))

        saver.save(sess,FLAGS.log_dir+'/model.ckpt',global_step=step)
      coord.request_stop()
      coord.join(threads)







def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
    tf.app.run()
