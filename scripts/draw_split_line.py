import cv2
import os
import sys
import xml.etree.ElementTree as ET

OUTP_DIR = '/home/fangsh/tianchi/tianchi_dataset/data_deal_3x3/xiaci_with_bbox_and_line'
ROOT_DIR = '/home/fangsh/tianchi/tianchi_dataset/draw_outp/drow_new'
#XML_ROOT = '/home/fangsh/tianchi/tianchi_dataset/data_megred/train'



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
        '''
        if cur_cls == '正常':
            cur_name = cur_img_dir.split('/')[-1][0:-4]
            image = cv2.imread(cur_img_dir)
            for i in range(5):
                for j in range(5):
                    crop_img = image[j*384:(j+1)*384,i*512:(i+1)*512]
                    cur_outp_dir = os.path.join(OUTP_DIR,cur_cls,cur_name)
                    crop_img_name = cur_name+'_%d_%d.jpg'%(i,j)
                    if not os.path.exists(cur_outp_dir):
                        os.makedirs(cur_outp_dir)

                    cv2.imwrite(os.path.join(cur_outp_dir,crop_img_name),crop_img)
       '''
        if not cur_cls=='正常':
            cur_name = cur_img_dir.split('/')[-1][0:-4]
            cls = cur_img_dir.split('/')[-2]
            image = cv2.imread(cur_img_dir)

            #xml_dir = os.path.join(XML_ROOT,cls,cur_name+'.xml')

            #tree = ET.parse(xml_dir)
            #root = tree.getroot()
            #bboxes = []

            #for obj in root.findall('object'):

                #bbox = obj.find('bndbox')
                #bboxes.append((int(bbox[0].text),
                              # int(bbox[1].text),
                               #int(bbox[2].text),
                              # int(bbox[3].text)))
            for i in range(2):
                image[:,(i+1)*853-2:(i+1)*853+2] = [0,255,0]
                image[(i+1)*640-2:(i+1)*640+2,:] = [0,255,0]
            cur_outp_dir = os.path.join(OUTP_DIR,cur_cls)
            crop_img_name = cur_name+'.jpg'
            if not os.path.exists(cur_outp_dir):
                os.makedirs(cur_outp_dir)

                #a1 = list(range(j*384,(j+1)*384))
                #b1 = list(range(i*512,(i+1)*512))
                #save_flag = estimate_cur_img_with_bbox(a1,b1,bboxes)
                #if save_flag:
            cv2.imwrite(os.path.join(cur_outp_dir,crop_img_name),image)



if __name__ == '__main__':
    main()
