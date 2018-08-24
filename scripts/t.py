import tensorflow as tf
from nets import vgg
#import data_input
from preprocessing.vgg_preprocessing import preprocess_image
slim = tf.contrib.slim



def preprocess_for_eval(image, height, width,
                        central_fraction=None, scope=None):
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def main():

    checkpoint_path = '/home/fangsh/tianchi/competition/slim/tianchi/train_dir'
    image_dir = '/home/fangsh/J01_2018.06.13 13_17_04_2_2.jpg'
    size = vgg.vgg_16.default_image_size
    with tf.variable_scope('input'):
        image = tf.gfile.FastGFile(image_dir,'rb').read()
        image = tf.image.decode_jpeg(image)
        image = preprocess_image(image,size,size)
        print(image)
        #image = tf.expand_dims(image, 0)
        image = tf.reshape(image,[-1,224,224,3])
    with slim.arg_scope(vgg.vgg_arg_scope()):
        outputs,end_points = vgg.vgg_16(image,is_training=False,num_classes=2)
        outputs = tf.squeeze(outputs)

    with tf.Session() as sess:
        #tf.global_variables_initializer().run()
        #tf.local_variables_initializer().run()
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)

        out = sess.run([outputs])
        print(out)

main()

