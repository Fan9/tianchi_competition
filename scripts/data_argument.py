#coding=utf-8
import tensorflow as tf
import numpy as np
import glob
import os
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt




def data_aug(image,):

    color_ordering = np.random.randint(4)

    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=16. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=16. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.1)
    elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.1)
        image = tf.image.random_brightness(image, max_delta=16. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.1)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=16. / 255.)

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return (image)


def get_image_dirs(root_dir):


    file_dir = []
    images_dir = []
    for root, sub, files in os.walk(root_dir):
        for file in files:
            file_dir.append(os.path.join(root, file))

    for f in file_dir:
        if f[-4:] == '.jpg':
            images_dir.append(f)
    return images_dir


def main(root_dir,outp_dir):



    img_raw = tf.placeholder(tf.string,shape=[1],name='input')
    img_raw = tf.squeeze(img_raw)
    img = tf.image.decode_jpeg(img_raw)
    #img = tf.image.convert_image_dtype(img,tf.float32)
    image = data_aug(img)
    #image = tf.image.convert_image_dtype(image,tf.uint8)

    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()


        #plt.imshow(image.eval())

        # norm_images_dir = '/home/fangsh/tianchi/tianchi_dataset/data_deal_3x3/selected_crop_xiaci/norm'
        # norm_images_data = os.listdir(norm_images_dir)
        # #images_dir = get_image_dirs(root_dir)
        # for image_name in tqdm(norm_images_data):
        #     cur_image_path = os.path.join(norm_images_dir,image_name)
        #     #cur_cls = img_dir.split('/')[-2]
        #     # print(cur_cls)
        #     #cur_name = img_dir.split('/')[-1][:-4]
        #
        #     # image_dir = tf.placeholder(tf.string,[1])
        #     img1 = tf.gfile.FastGFile(cur_image_path, 'rb').read()
        #
        #     cur_outp_dir = os.path.join(outp_dir,'norm',image_name)
        #     image1 = sess.run(image,feed_dict={img_raw:img1})
        #     cv2.imwrite(cur_outp_dir,image1)
        #             #plt.imsave('/home/fangsh/test/test1/'+cur_name+'_%d.jpg'%i,image.eval())

        xiaci_images_dir = '/home/fangsh/tianchi/tianchi_dataset/data_deal_3x3/selected_crop_xiaci/xiaci_randaom_crop/xiaci'
        xiaci_images_data = os.listdir(xiaci_images_dir)
        # images_dir = get_image_dirs(root_dir)
        for image_name in tqdm(xiaci_images_data):
            cur_image_path = os.path.join(xiaci_images_dir, image_name)
            # cur_cls = img_dir.split('/')[-2]
            # print(cur_cls)
            # cur_name = img_dir.split('/')[-1][:-4]

            # image_dir = tf.placeholder(tf.string,[1])
            img1 = tf.gfile.FastGFile(cur_image_path, 'rb').read()

            cur_outp_dir = os.path.join(outp_dir, 'xiaci', image_name.split(' ')[0]+'_'+image_name.split(' ')[1])
            image1 = sess.run(image, feed_dict={img_raw: img1})
            cv2.imwrite(cur_outp_dir, image1)



if __name__ == '__main__':
    root_dir = '/home/fangsh/tianchi/tianchi_dataset/data_megred/train'
    outp_dir = '/media/fangsh/data/tianchi/augment_data/3x3'
    main(root_dir,outp_dir)




