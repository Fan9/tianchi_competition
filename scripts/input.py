import tensorflow as tf
import os


NUM_CLASSES = 10
NUM_EXAMPLES_FOR_TRAIN = 1415
NUM_EXAMPLES_FOR_EVAL = 607
INPUT_SIZE=(400,500)



def _generate_image_and_label_batch(image,label,min_queue_examples,
                                    batch_size,shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images,labels = tf.train.shuffle_batch([image,label],
                                               batch_size=batch_size,
                                               min_after_dequeue=min_queue_examples,
                                               capacity=min_queue_examples+3*batch_size,
                                               num_threads=num_preprocess_threads)
    else:
        images,labels = tf.train.batch([image,label],
                                       batch_size=batch_size,
                                       num_threads=min_queue_examples,
                                       capacity=min_queue_examples+batch_size*3)
    return images,labels


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0,scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    fast_mode = True
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_image_for_train(image,height,width):

    with tf.name_scope('distorted_image'):
        distorted_image = apply_with_random_selector(
            image,
            lambda x, method: tf.image.resize_images(x, [height, width], method=method),
            num_cases=num_resize_cases)

        tf.summary.image('cropped_resized_image',
                         tf.expand_dims(distorted_image, 0))

        distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = apply_with_random_selector(
            distorted_image,
            lambda x, ordering: distort_color(x, ordering),
            num_cases=4)

        tf.summary.image('final_distorted_image',
                         tf.expand_dims(distorted_image, 0))
        distorted_image = tf.subtract(distorted_image, 0.5)
        distorted_image = tf.multiply(distorted_image, 2.0)

        return distorted_image


def preprocess_image_for_test(image,height,width):
    with tf.name_scope('eval_image'):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width],
                                     align_corners=False)
        image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)

        return image

def inputs(is_traing,data_dir,batch_size):
    if is_traing:
        file_dir = os.path.join(data_dir,'train.tfrecord')
        num_examples_per_epoch = NUM_EXAMPLES_FOR_TRAIN
    else:
        file_dir = os.path.join(data_dir,'test.tfrecord')
        num_examples_per_epoch = NUM_EXAMPLES_FOR_EVAL

    if not tf.gfile.Exists(file_dir):
        raise ValueError('Failed to find file%s'%file_dir)
    with tf.name_scope('input'):

        file_queue = tf.train.string_input_producer([file_dir])
        reader = tf.TFRecordReader()
        _,serialized = reader.read(file_queue)
        feature = tf.parse_single_example(serialized=serialized,
                                          features={
                                              'image/encoded': tf.FixedLenFeature([], tf.string),
                                              'image/format': tf.FixedLenFeature([], tf.string),
                                              'image/shape': tf.FixedLenFeature([3], tf.int64),
                                              'label': tf.FixedLenFeature([], tf.int64),
                                              'object/name': tf.VarLenFeature(tf.string),
                                              'object/truncated': tf.VarLenFeature(tf.int64),
                                              'object/difficult': tf.VarLenFeature(tf.int64),
                                              'object/bbox/xmin': tf.VarLenFeature(tf.float32),
                                              'object/bbox/ymin': tf.VarLenFeature(tf.float32),
                                              'object/bbox/xmax': tf.VarLenFeature(tf.float32),
                                              'object/bbox/ymax': tf.VarLenFeature(tf.float32)
                                                })
        image_raw,label = feature['image/encoded'], feature['label']
        image_data = tf.image.decode_jpeg(image_raw)
        image_data = tf.image.convert_image_dtype(image_data,tf.float32)

        if is_traing:
            pre_processed_image = preprocess_image_for_train(image_data,height=INPUT_SIZE[0],width=INPUT_SIZE[1])
        else:
            pre_processed_image = preprocess_image_for_test(image_data,height=INPUT_SIZE[0],width=INPUT_SIZE[1])


        label = tf.cast(label, tf.int32)



        #reshape_image = tf.reshape(image,[1920,2560,3])
        #tf.decode_raw()解码二进制数据返回的是dim=1的list,需要用tf.reshpe()变成dim=3的图像形式
        #tf.image.decode_jpeg()解码二进制返回的直接就是jpeg格式编码的图像

        #resized_image = tf.image.resize_images(float_image,224,224)
        #standard_image = tf.image.per_image_standardization(resized_image)

        #standard_image.setshape(shape)
        #label.setshape([1])

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(pre_processed_image,label,min_queue_examples,
                                           batch_size,shuffle=True)






