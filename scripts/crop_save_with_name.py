import cv2
import os
import sys
from tqdm import tqdm
import xml.etree.ElementTree as ET

OUTP_DIR = '/media/fangsh/data/tianchi/data/5x5'
ROOT_DIR = '/home/fangsh/tianchi/tianchi_dataset/data_megred/train'
INDEX_DIR = '/home/fangsh/tianchi/tianchi_dataset/data_deal_3x3/selected_crop_xiaci_with_bbox'


def estimate_cur_img_with_bbox(a1,b1,bboxes):
    '''

    :param a1: image height range
    :param b1: image width range
    :param bboxes: bboxes
    :return: save flag
    '''
    save_flag = False
    for box in bboxes:
        a2 = list(range(box[1],box[3]))
        b2 = list(range(box[0],box[2]))
        tmp1 = [var for var in a1 if var in a2]
        tmp2 = [var for var in b1 if var in b2]
        if tmp1 and tmp2:
            save_flag = True
            break
    return save_flag

def main():

    cls_info = []
    file_dir = []


    for root, sub, files in os.walk(ROOT_DIR):
        cls_info.extend(sub)
        for file in files:
            file_dir.append(os.path.join(root, file))
    #print(cls_info)

    image_dir = []
    for i in file_dir:
        #print(i.split('/')[-1][-4:])
        if i.split('/')[-1][-4:] == '.jpg':
            image_dir.append(i)

    for idx,cur_img_dir in enumerate(image_dir):
        sys.stdout.write('\r >> Convert to %s   %d/%d' % (i, idx + 1, len(image_dir)))

        cur_cls = cur_img_dir.split('/')[-2]

        if cur_cls == '正常':
            cur_name = cur_img_dir.split('/')[-1][0:-4]
            image = cv2.imread(cur_img_dir)
            for i in range(5):
                for j in range(5):
                    crop_img = image[i*384:(i+1)*384,j*512:(j+1)*512]
                    cur_outp_dir = os.path.join(OUTP_DIR,'norm')
                    crop_img_name = cur_name+'_%d_%d.jpg'%(i,j)
                    if not os.path.exists(cur_outp_dir):
                        os.makedirs(cur_outp_dir)

                    cv2.imwrite(os.path.join(cur_outp_dir,crop_img_name),crop_img)
        else:
            cur_name = cur_img_dir.split('/')[-1][0:-4]
            cls = cur_img_dir.split('/')[-2]
            image = cv2.imread(cur_img_dir)
            '''
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
            '''
            #name_need_save = os.listdir(INDEX_DIR)
            for i in range(5):
                for j in range(5):
                    crop_img = image[i*384:(i+1)*384,j*512:(j+1)*512]
                    cur_outp_dir = os.path.join(OUTP_DIR,'xiaci')
                    crop_img_name = cur_name+'_%d_%d.jpg'%(i,j)
                    if not os.path.exists(cur_outp_dir):
                        os.makedirs(cur_outp_dir)

                    #a1 = list(range(j*384,(j+1)*384))
                    #b1 = list(range(i*512,(i+1)*512))
                    
                    
                    #if crop_img_name in name_need_save:
                    cv2.imwrite(os.path.join(cur_outp_dir,crop_img_name),crop_img)



if __name__ == '__main__':
    main()