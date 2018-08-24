import tensorflow as tf
from tqdm import tqdm
#from convert_common import int64_feature,float_feature,bytes_feature
from convert_common import _convert_to_example
import os


def main():
    root_dir = '/media/fangsh/data/tianchi/augment_data/test'
    outp_dir = '/media/fangsh/data/tianchi/data/tfrecord/test.tfrecord'

    with tf.python_io.TFRecordWriter(outp_dir) as writer:

        # convert norm data
        norm_data_dir = os.path.join(root_dir,'norm')
        norm_images = os.listdir(norm_data_dir)
        norm_images.sort()

        with open('/media/fangsh/data/tianchi/data/tfrecord/testlist.txt','a') as f:

            for image in tqdm(norm_images):
                f.write(image+'\n')
                cur_image_path = os.path.join(norm_data_dir,image)
                img_raw = tf.gfile.FastGFile(cur_image_path,'rb').read()
                label = 0
                example = _convert_to_example(img_raw,label)
                writer.write(example.SerializeToString())


            # convert xiaci data
            xiaci_data_dir = os.path.join(root_dir, 'xiaci')
            xiaci_images = os.listdir(xiaci_data_dir)
            xiaci_images.sort()

            for image in tqdm(xiaci_images):
                f.write(image+'\n')
                cur_image_path = os.path.join(xiaci_data_dir, image)
                img_raw = tf.gfile.FastGFile(cur_image_path, 'rb').read()
                label = 1
                example = _convert_to_example(img_raw, label)
                writer.write(example.SerializeToString())

        print('Finished convert tianchi dataset!')

if __name__ == '__main__':
    main()