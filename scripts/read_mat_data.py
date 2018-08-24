import scipy.io as sio
import numpy as np
import h5py
import os
from tqdm import tqdm

#path = '/media/fangsh/My Passport/vgg_features/features/conv6_1/val/J01_2018.06.13_13_24_39_blob_0.mat'
#mat = sio.loadmat(path)
#data = mat['data']
#print(np.squeeze(data,3).shape)


def merged_feature():
    h5file_dir = '/media/fangsh/data/tianchi/test/test.h5'
    mat_root_dir = '/media/fangsh/My Passport/testb'
    mat_crop_dir = '/media/fangsh/My Passport/features/layer_512_3_conv2/norm_out'
    name_dir = '/media/fangsh/data/tianchi/test/test_ronghe.txt'
    outp_dir = '/media/fangsh/data/tianchi/test/merged_test.h5'

    names = []
    with open(name_dir,'r') as f:
        for line in f.readlines():
            names.append(line)
    #print(names)

    with h5py.File(h5file_dir) as h5:
        train = np.array(h5['test'])
        #label = np.array(h5['label'])

    merged_data = np.zeros([len(train), 35, 35, 768])
    for idx in tqdm(range(len(train))):
        merged_data[idx,:,:,0:512] = train[idx]
        cur_name = names[idx]

        cur_mat_dir = os.path.join(mat_root_dir,
                                   cur_name.split(' ')[0]+'_'+cur_name.split(' ')[1].strip()[0:-4]+'_blob_0.mat')
        mat = sio.loadmat(cur_mat_dir)
        data = np.squeeze(mat['data'],3)
        merged_data[idx,:,:,512:768] = data

        # for i in range(5):
        #     for j in range(5):
        #         cur_crop_mat_dir = os.path.join(mat_crop_dir,
        #                                         cur_name.split(' ')[0]+'_'+cur_name.split(' ')[1].strip()[0:-4] \
        #                                         +'_%d_%d_blob_0.mat'%(i,j))
        #
        #         mat = sio.loadmat(cur_crop_mat_dir)
        #         data = np.squeeze(mat['data'],3)
        #         merged_data[idx,i*7:(i+1)*7,j*7:(j+1)*7,768:1280] = data

    with h5py.File(outp_dir) as h5:
        h5.create_dataset('test',data=merged_data)
        #h5.create_dataset('label',data=label)


if __name__ == '__main__':
    h5file_dir = '/home/fangsh/test/vgg_16_1.h5'
    mat_root_dir = '/media/fangsh/My Passport/vgg_features/features/conv6_1/train'
    mat_crop_dir = '/media/fangsh/My Passport/features/layer_512_3_conv2/norm_out'
    name_dir = '/home/fangsh/test/name0731.txt'
    outp_dir = '/media/fangsh/data/tianchi/feature_map/merged_train.h5'
    merged_feature()
