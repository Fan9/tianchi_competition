import tensorflow as tf
import cv2
import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import sys


root_dir = '/home/fangsh/projects/tianchi_cometition/data/chusha/'
image_name = 'J01_2018.06.19 08_58_24.jpg'
def draw_bbox_per_image(root_dir,image_name,outp_dir,cur_cls):

    image_dir = os.path.join(root_dir,cur_cls,image_name)

    img = cv2.imread(image_dir)
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    xml_dir = os.path.join(root_dir,cur_cls,image_name[:-4]+'.xml')
    tree = ET.parse(xml_dir)
    root = tree.getroot()
    size = root.find('size')
    shape=[int(size.find('height').text),
           int(size.find('width').text),
           int(size.find('depth').text)]

    object_name = []
    truncated = []
    difficult = []
    bboxes = []

    for obj in root.findall('object'):
        #object_name.append(obj.find('name').text.encode('utf-8'))
        object_name.append(obj.find('name').text)
        truncated.append(int(obj.find('truncated').text))
        difficult.append(int(obj.find('difficult').text))
        bbox = obj.find('bndbox')
        bboxes.append((int(bbox[0].text),
                       int(bbox[1].text),
                       int(bbox[2].text),
                       int(bbox[3].text)))


    dif_cls = [object_name[0]]
    if object_name[1:]:
        for idx,c in enumerate(object_name[1:]):
            if c not in dif_cls:
                dif_cls.append(c)

    #colors = [(0,245,255),(255,228,181),(255,69,0),(144,238,144)]
    colors = [(84,255,159),(0,191,255),(255,106,106),(255,215,0)]
    for idx,name in enumerate(object_name):
        color = colors[dif_cls.index(name)]

        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img,cls,(bboxes[idx][0],bboxes[idx][1]),font,\
                   # 3,color,2)

        font = ImageFont.truetype('/home/fangsh/anaconda3/envs/tensorflow3/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/simhei.ttf', 50)
        draw = ImageDraw.Draw(img_PIL)
        draw.text((bboxes[idx][0],bboxes[idx][3]),name,font=font,
                  fill=(color[2],color[1],color[0]))
        
    img_cv = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    for idx, cls in enumerate(object_name):
        color = colors[dif_cls.index(cls)]
        cv2.rectangle(img_cv,(bboxes[idx][0],bboxes[idx][1]),
                      (bboxes[idx][2],bboxes[idx][3]),color,4)



    #cv2.imshow('show',img_cv)
    #cv2.waitKey()
    cur_outp_dir = os.path.join(outp_dir,cur_cls)

    if not os.path.exists(cur_outp_dir):
        os.makedirs(cur_outp_dir)
    cv2.imwrite(os.path.join(cur_outp_dir,image_name),img_cv)
    #cv2.imwrite('/home/fangsh/projects/tianchi_cometition/data/show_label/rr12.jpg',img_cv)
def save_draw_bbox_image(root_dir,outp_dir):
    
    classes = os.listdir(root_dir)
    for lab,cur_cls in enumerate(classes):

        sys.stdout.write('\n')
        #dict to save classes info
        class_dict={}
        class_dict[lab] = cur_cls

        cur_cls_dir = os.path.join(root_dir,cur_cls,'*.jpg')
        images_dir = glob.glob(cur_cls_dir)

        if not cur_cls == '正常':
            for idx,image_dir in enumerate(images_dir):

                format_str = '\r  >> Draw the cls: %s%s%d/%d'
                sys.stdout.write(format_str % (cur_cls, (10 - len(cur_cls)) * ' ', idx + 1, len(images_dir)))
                sys.stdout.flush()

                image_name = image_dir.split('/')[-1]
                draw_bbox_per_image(root_dir,image_name,outp_dir,cur_cls)

    print("\nFinished draw image bbox!")
    

if __name__ == '__main__':
    root_dir = '/home/fangsh/tianchi/tianchi_dataset/data_megred/train'
    outp_dir = '/home/fangsh/tianchi/tianchi_dataset/draw_outp/drow_new'
    save_draw_bbox_image(root_dir=root_dir,outp_dir=outp_dir)

