import cv2
import os
import sys
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random

OUTP_DIR = '/home/fangsh/tianchi/tianchi_dataset/data_deal_3x3/selected_crop_xiaci/xiaci_randaom_crop'
ROOT_DIR = '/home/fangsh/tianchi/tianchi_dataset/raw_merged'
INDEX_DIR = '/home/fangsh/tianchi/tianchi_dataset/data_deal_3x3/selected_crop_xiaci_with_bbox'
XML_ROOT = '/home/fangsh/tianchi/tianchi_dataset/raw_merged'

def random_crop(image_index,bboxes,cur_output,image):
    '''

    :param image_index: a list[image_x1,image_y1,image_x2,image_y2]
    :param b1: image width range
    :param bboxes: bboxes
    :return: save flag
    '''
    #save_flag = False

    a1 = list(range(image_index[1], image_index[3]))
    b1 = list(range(image_index[0], image_index[2]))
    for box in bboxes:
        a2 = list(range(box[1],box[3]))
        b2 = list(range(box[0],box[2]))
        tmp1 = [var for var in a1 if var in a2]
        tmp2 = [var for var in b1 if var in b2]
        if tmp1 and tmp2:

            #bbox location relative to crop_image
            crop_image_bbox = [max(box[0],image_index[0]),max(box[1],image_index[1]),
                                   min(box[2],image_index[2]),min(box[3],image_index[3])]
           # print(crop_image_bbox)
            x_random_scope = [max(0,crop_image_bbox[2]-853),min(crop_image_bbox[0],2560-853)]
            y_random_scope = [max(0,crop_image_bbox[3]-640),min(crop_image_bbox[1],1920-640)]
            #print(x_random_scope,y_random_scope)
            for i1 in range(8):
                x1 = random.randint(x_random_scope[0],x_random_scope[1])
                y1 = random.randint(y_random_scope[0],y_random_scope[1])
                crop_image = image[y1:y1+640,x1:x1+853]
                cur_outp_dir1 = cur_output+'_%d_1.jpg'%i1
                cv2.imwrite(cur_outp_dir1,crop_image)






def main():
    cls_info = []
    file_dir = []

    for root, sub, files in os.walk(ROOT_DIR):
        cls_info.extend(sub)
        for file in files:
            file_dir.append(os.path.join(root, file))
    # print(cls_info)

    image_dir = []
    for i in file_dir:
        # print(i.split('/')[-1][-4:])
        if i.split('/')[-1][-4:] == '.jpg':
            image_dir.append(i)

    for idx, cur_img_dir in enumerate(tqdm(image_dir)):
        #sys.stdout.write('\r >> Convert to %s   %d/%d' % (i, idx + 1, len(image_dir)))

        cur_cls = cur_img_dir.split('/')[-2]

        if not cur_cls == '正常':

            cur_name = cur_img_dir.split('/')[-1][0:-4]
            cls = cur_img_dir.split('/')[-2]
            image = cv2.imread(cur_img_dir)

            #sparse xml file
            xml_dir = os.path.join(XML_ROOT,cls,cur_name+'.xml')

            tree = ET.parse(xml_dir)
            root = tree.getroot()
            bboxes = []

            for obj in root.findall('object'):

                bbox = obj.find('bndbox')
                bboxes.append((int(bbox[0].text),
                               int(bbox[1].text),
                               int(bbox[2].text),
                               int(bbox[3].text)))

            name_need_save = os.listdir(INDEX_DIR)
            for i in range(3):
                for j in range(3):
                    #crop_img = image[j * 640:(j + 1) * 640, i * 853:(i + 1) * 853]
                    cur_outp_dir = os.path.join(OUTP_DIR, 'xiaci')
                    crop_img_name = cur_name + '_%d_%d.jpg' % (i, j)
                    if not os.path.exists(cur_outp_dir):
                        os.makedirs(cur_outp_dir)

                    image_index = [i * 853,j * 640,(i + 1) * 853,(j + 1) * 640]
                    cur_output = os.path.join(cur_outp_dir,crop_img_name)
                    if crop_img_name in name_need_save:
                        random_crop(image_index,bboxes,cur_output[0:-4],image)


if __name__ == '__main__':
    main()