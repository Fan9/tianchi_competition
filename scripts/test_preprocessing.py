#coding=utf-8
import tensorflow as tf
from matplotlib import pyplot as plt


image_raw = tf.gfile.FastGFile('/home/fangsh/01.jpg','rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw)
    float_image = tf.cast(img_data,tf.float32)
    #tf.cast(img_data,tf.float32)仅仅是将整形转为浮点型

    #print(float_image.eval())
    #print(img_data.eval())
    img_data = tf.image.convert_image_dtype(img_data,dtype=tf.float32)
    #tf.image.convert_image_dtype(,tf.float32)将图像转为0.0-1.0范围的.8f的浮点型
    #在所有的图像预处理前先将图像数据转化为实数类型

    #print(img_data.eval())
    resized1 = tf.image.resize_images(img_data,[300,300],method=0)
    resized2 = tf.image.resize_images(img_data, [300, 300], method=1)
    resized3 = tf.image.resize_images(img_data, [300, 300], method=2)
    resized4 = tf.image.resize_images(img_data, [300, 300], method=3)

    '''
    fig,axes = plt.subplots(2,3)
    [a,b,c],[d,e,f] = axes
    a.imshow(img_data.eval())
    b.imshow(resized1.eval())
    c.imshow(resized2.eval())
    d.imshow(resized3.eval())
    e.imshow(resized4.eval())
    #plt.imshow(resized.eval())
    plt.show()
    '''

    adjusted = tf.image.per_image_standardization(img_data)
    #将图像三维矩阵的数字变为均值是0，方差是1

    #fig, axes = plt.subplots(1,2)
    #a,b = axes
    #a.imshow(img_data.eval())
    #b.imshow(adjusted.eval())
    #plt.show()

    img_data = tf.image.resize_images(img_data,[300,300],method=1)
    batched = tf.expand_dims(
        img_data, 0)
    boxes = tf.constant([[[0.6, 0.01, 0.7, 0.9]]])
    result = tf.image.draw_bounding_boxes(batched, boxes)
    #result.eval()
    img1 = tf.reshape(result,[300,300,3])
    #tf.image.draw_bounding_boxes()要求batch数据，所以在使用之前先将图像数据扩展一维
    #plt.imshow(img1.eval())
    #plt.show()

    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img_data), bounding_boxes=boxes, min_object_covered=0.4)
    batched1 = tf.expand_dims(
        img_data, 0)

    image_with_box = tf.image.draw_bounding_boxes(batched1, bbox_for_draw)
    distorted = tf.slice(img_data, begin, size)

    img_for_show = tf.reshape(image_with_box,[300,300,3])

    fig, axes = plt.subplots(1,2)
    a,b = axes
    a.imshow(img_for_show.eval())
    b.imshow(distorted.eval())
    plt.show()


