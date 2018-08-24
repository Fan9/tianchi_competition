#coding=utf-8
import os
import tqdm


def choose_varification_date():
    pass

def get_images_dir(root_dir):
    file_dir = []
    norm_images_dir = []
    xiaci_images_dir = []

    for root,subdocs,file in os.walk(root_dir):
        for subdoc in subdocs:
            file_dir.append(os.path.join(root,file))

    for file in file_dir:
        if file[1:-4] == '.jpg':
            if file.split('/')[-2]=='正常':

                norm_images_dir.append(file)
            else:
                xiaci_images_dir.append(file)

    return norm_images_dir,xiaci_images_dir















if __name__ == '__main__':
    outp_dir= '/home/fangsh/tianchi/tianchi_dataset/varification'
    root_dir = ''
    main()