import tensorflow as tf
import os


NUM_CLASSES = 10
NUM_EXAMPLES_FOR_TRAIN = 2000
NUM_EXAMPLES_FOR_EVAL = 500

def _generate_image_and_label_batch(image,label,min_queue_examples,
                                    batch_size,shuffle):
    print(min_queue_examples)
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
                                       num_threads=num_preprocess_threads,
                                       capacity=min_queue_examples+batch_size*3)
    return images,labels

def inputs(is_train,data_dir,batch_size):
    if is_train:
        file_dir = os.path.join(data_dir,'train.tfrecord')
        num_examples_per_epoch = NUM_EXAMPLES_FOR_TRAIN

    else:
        file_dir = os.path.join(data_dir, 'test.tfrecord')
        num_examples_per_epoch = NUM_EXAMPLES_FOR_EVAL

    if not tf.gfile.Exists(file_dir):
        raise ValueError('Failed to find file%s'%file_dir)
    with tf.name_scope('input'):

        file_queue = tf.train.string_input_producer([file_dir],num_epochs=50)
        reader = tf.TFRecordReader()
        _,serialized = reader.read(file_queue)
        feature = tf.parse_single_example(serialized=serialized,
                                          features={
                                              'image/shape': tf.FixedLenFeature([3],tf.int64),
                                              'label': tf.FixedLenFeature([],tf.int64),
                                              'image_raw':tf.FixedLenFeature([],tf.string),
                                              'object/name': tf.VarLenFeature(tf.string),
                                              'object/truncated':tf.VarLenFeature(tf.int64),
                                              'object/difficult':tf.VarLenFeature(tf.int64),
                                              'object/xmin':tf.VarLenFeature(tf.float32),
                                              'object/ymin':tf.VarLenFeature(tf.float32),
                                              'object/xmax':tf.VarLenFeature(tf.float32),
                                              'object/ymax':tf.VarLenFeature(tf.float32)
                                                })
        shape,label = feature['image/shape'], feature['label']
        image_raw = feature['image_raw']

        # process lael
        #change 42 classes to 2 classes
        #label = 0 if label==33 else 1
        label = tf.cast(label, tf.int32)

        # process image
        image = tf.image.decode_jpeg(image_raw)
        float_image = tf.cast(image, tf.float32)
        float_image = tf.reshape(float_image,[1920,2560,3])

        #reshape_image = tf.reshape(image,[1920,2560,3])
        #tf.decode_raw()解码二进制数据返回的是dim=1的list,需要用tf.reshpe()变成dim=3的图像形式
        #tf.image.decode_jpeg()解码二进制返回的直接就是jpeg格式编码的图像

        resized_image = tf.image.resize_images(float_image,[224,224])
        standard_image = tf.image.per_image_standardization(resized_image)
        standard_image = tf.reshape(standard_image,[224,224,3])

        #standard_image.setshape(shape)
        #label.setshape([1])

        min_fraction_of_examples_in_queue = 0.2
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(standard_image,label,min_queue_examples,
                                           batch_size,shuffle=True)

def distorted_inputs(data_dir,batch_size):

    file_dir = os.path.join(data_dir,'train.tfrecord')
    num_examples_per_epoch = NUM_EXAMPLES_FOR_TRAIN
    if not tf.gfile.Exists(file_dir):
        raise ValueError('Failed to find file%s'%file_dir)
    with tf.name_scope('input'):

        file_queue = tf.train.string_input_producer([file_dir],num_epochs=50)
        reader = tf.TFRecordReader()
        _,serialized = reader.read(file_queue)
        feature = tf.parse_single_example(serialized=serialized,
                                          features={
                                              'image/shape': tf.FixedLenFeature([3],tf.int64),
                                              'label': tf.FixedLenFeature([],tf.int64),
                                              'image_raw':tf.FixedLenFeature([],tf.string),
                                              'object/name': tf.VarLenFeature(tf.string),
                                              'object/truncated':tf.VarLenFeature(tf.int64),
                                              'object/difficult':tf.VarLenFeature(tf.int64),
                                              'object/xmin':tf.VarLenFeature(tf.float32),
                                              'object/ymin':tf.VarLenFeature(tf.float32),
                                              'object/xmax':tf.VarLenFeature(tf.float32),
                                              'object/ymax':tf.VarLenFeature(tf.float32)
                                                })
        shape,label = feature['image/shape'], feature['label']
        image_raw = feature['image_raw']

        # process lael
        # change 42 classes to 2 classes
        #label = 0 if label == [33] else 1
        label = tf.cast(label, tf.int32)

        # process image
        image = tf.image.decode_jpeg(image_raw)
        float_image = tf.cast(image, tf.float32)
        float_image = tf.reshape(float_image, [224, 224, 3])

        # data argumentation

        # Randomly crop a [height, width] section of the image.
        #distorted_image = tf.random_crop(image, shape)
        resized_image = tf.image.resize_images(float_image, (224, 224))

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(resized_image)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        # NOTE: since per_image_standardization zeros the mean and makes
        # the stddev unit, this likely has no effect see tensorflow#1458.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)


        standard_image = tf.image.per_image_standardization(distorted_image)

        #standard_image.setshape(shape)
        #label.setshape([1])

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(standard_image,label,min_queue_examples,
                                           batch_size,shuffle=True)




