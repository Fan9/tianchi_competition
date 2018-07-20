import tensorflow as tf
import cv2
import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import numpy as np


root_dir = '/home/fangsh/projects/tianchi_cometition/data/classes/chusha/'
image_name = 'J01_2018.06.23 09_16_50.jpg'
def show_per_image_label(image_name,):

    image_dir = os.path.join(root_dir,image_name)
    img = cv2.imread(image_dir)
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    xml_dir = os.path.join(root_dir,image_name[:-4]+'.xml')
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

    print(object_name)


    dif_cls = object_name[0]
    if object_name[1:]:
        for idx,c in enumerate(object_name[1:]):
            if c not in dif_cls:
                dif_cls.append(c)

    colors = [(255,228,181),(144,238,144),(255,69,0),(0,245,255)]
    for idx,cls in enumerate(object_name):
        color = colors[dif_cls.index(cls)]

        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img,cls,(bboxes[idx][0],bboxes[idx][1]),font,\
                   # 3,color,2)

        font = ImageFont.truetype('/home/fangsh/Downloads/simheittf/simhei.ttf', 80)
        draw = ImageDraw.Draw(img_PIL)
        draw.text((bboxes[idx][0],bboxes[idx][3]),cls,font=font,
                  fill=(color[2],color[1],color[0]))

    img_cv = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    for idx, cls in enumerate(object_name):
        color = colors[dif_cls.index(cls)]
        cv2.rectangle(img_cv,(bboxes[idx][0],bboxes[idx][1]),
                      (bboxes[idx][2],bboxes[idx][3]),color,4)




    #cv2.imshow('show',img_cv)
    #cv2.waitKey()
    cv2.imwrite('/home/fangsh/projects/tianchi_cometition/data/show_label/rr12.jpg',img_cv)

show_per_image_label(image_name)