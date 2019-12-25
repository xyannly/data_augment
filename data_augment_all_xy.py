'''
Readme:
    数据增强脚本
    用于训练数据存在数据不均衡问题
    通过对数量较少的图片进行镜像、旋转、随机裁剪等操作，获得更多的训练数据
    输入为图片路径，jpg和xml存在同一文件夹下
'''
import os
import cv2
import sys
import json
import math
import random
import logging
import numpy as np


import xml.etree.ElementTree as ET
import shutil
import re

logging.getLogger().setLevel(logging.INFO)
#IMAGE_PATH = "./images/"
#LABEL_PATH = "./labels/"

def create_mirror(label_name,img,img_width,img_height):
    postfix = label_name.split('.')[0]
    output_path = postfix + "_mirror.jpg"
    label_output_path = postfix + '_mirror.xml'
    print("create mirror: " + postfix+'.jpg')

    #创建镜像img的xml
    shutil.copy(label_name,label_output_path)

    #xml修改
    tree = ET.parse(label_output_path)
    root = tree.getroot()
    for object in root.iter('object'):
        xmin = int(object.find('bndbox/xmin').text)
        ymin = int(object.find('bndbox/ymin').text)
        xmax = int(object.find('bndbox/xmax').text)
        ymax = int(object.find('bndbox/ymax').text)
        x_center_new = img_width - (xmin + xmax) / 2
        xmin_new = int(x_center_new - (xmax - xmin) / 2)
        xmax_new = int(x_center_new + (xmax - xmin) / 2)
        object.find('bndbox/xmin').text = str(xmin_new)
        object.find('bndbox/xmax').text = str(xmax_new)
        tree.write(label_output_path, encoding="utf-8")

    #jpg修改
    mirror_img = np.zeros(img.shape, np.uint8)
    for i in range(img_height):
        for j in range(img_width):
            mirror_img[i, img_width - 1 - j] = img[i, j]  # left to right,correct，左右镜像
    #cv2.imwrite(output_path, mirror_img)
    cv2.imencode('.jpg', mirror_img)[1].tofile(output_path)
    print('create mirror compeleted for '+ output_path)





def random_cut(img, random_ratio,label_name,img_width,img_height):
    postfix = label_name.split('.')[0]
    output_path = postfix + "_cut_"+str(random_ratio)+".jpg"
    label_output_path = postfix + '_cut_'+str(random_ratio)+'.xml'
    # 创建镜像img的xml
    shutil.copy(label_name, label_output_path)
    tree = ET.parse(label_output_path)
    root = tree.getroot()
    if len(tree.findall('object'))>1:
        print(postfix+'.jpg has more than 1 object!')
        return 0
    for object in root.iter('object'):
        x1 = int(object.find('bndbox/xmin').text)
        y1 = int(object.find('bndbox/ymin').text)
        x2 = int(object.find('bndbox/xmax').text)
        y2 = int(object.find('bndbox/ymax').text)
        center_x = int((x1+x2)/2)
        center_y = int((y1+y2)/2)

        #get random location for 4 coordinate
        cut_x1 = random.randint(0, int(x1*random_ratio)+1)
        cut_y1 = random.randint(0, int(y1*random_ratio)+1)
        cut_x2 = img_width - random.randint(0, int((img_width-x2)*random_ratio)+1)
        cut_y2 = img_height - random.randint(0, int((img_height-y2)*random_ratio)+1)

        new_center_x = center_x - cut_x1
        new_center_y = center_y - cut_y1
        new_width = cut_x2 - cut_x1
        new_height = cut_y2 - cut_y1
        root.find('size/width').text = str(new_width)
        root.find('size/height').text = str(new_height)

        xmin_new = int(x1 - cut_x1)
        xmax_new = int(x2 - cut_x1)
        ymin_new = int(y1-cut_y1)
        ymax_new = int(y2-cut_y1)
        object.find('bndbox/xmin').text = str(xmin_new)
        object.find('bndbox/xmax').text = str(xmax_new)
        object.find('bndbox/ymin').text = str(ymin_new)
        object.find('bndbox/ymax').text = str(ymax_new)
        tree.write(label_output_path, encoding="utf-8")

        crop_img = img[cut_y1:cut_y2, cut_x1: cut_x2]
        cv2.imencode('.jpg', crop_img)[1].tofile(output_path)
        print('rand cut compeleted for '+postfix+'.jpg')

def random_rotate(img,label_name, rotate, img_width, img_height):
    postfix = label_name.split('.')[0]
    output_path = postfix + "_rotate_" + str(rotate) + ".jpg"
    label_output_path = postfix + '_rotate_' + str(rotate) + '.xml'

    rotate_img = np.zeros((img.shape[1], img.shape[0], img.shape[2]), np.uint8)

    shutil.copy(label_name, label_output_path)
    tree = ET.parse(label_output_path)
    root = tree.getroot()
    for object in root.iter('object'):
        xmin = int(object.find('bndbox/xmin').text)
        ymin = int(object.find('bndbox/ymin').text)
        xmax = int(object.find('bndbox/xmax').text)
        ymax = int(object.find('bndbox/ymax').text)

        if (rotate == 90):
            new_xmin = ymin
            new_ymin = img_width-xmax-1
            new_xmax = ymax
            new_ymax = img_width-xmin-1

            root.find('size/width').text = str(img_height)
            root.find('size/height').text = str(img_width)
        elif(rotate == 180):
            new_xmin = img_width - 1 - xmin
            new_ymin = img_width - 1 - ymin
            new_xmax = img_width - 1 - xmax
            new_ymax = img_width - 1 - ymax

        elif (rotate == 270):
            new_xmin = ymin
            new_ymin = img_width - 1 - xmax
            new_xmax = ymax
            new_ymax = img_width - 1 - xmin

            root.find('size/width').text = str(img_height)
            root.find('size/height').text = str(img_width)

        if new_xmin > new_xmax:
            new_xmax,new_xmin = new_xmin,new_xmax
            new_ymax,new_ymin = new_ymin,new_ymax
        object.find('bndbox/xmin').text = str(new_xmin)
        object.find('bndbox/xmax').text = str(new_xmax)
        object.find('bndbox/ymin').text = str(new_ymin)
        object.find('bndbox/ymax').text = str(new_ymax)
        tree.write(label_output_path, encoding="utf-8")
    for i in range(img_height):
        for j in range(img_width):
                if(rotate == 90):
                    rotate_img[img_width-1-j, i] = img[i,j]
                elif(rotate == 180):
                    rotate_img[img_height-1-i, img_width-1-j] = img[i,j]
                elif(rotate == 270):
                    rotate_img[j,img_height-1-i] = img[i,j]
                    #cv2.imwrite(output_path+"_3.jpg", rotate_img)
    cv2.imencode('.jpg', rotate_img)[1].tofile(output_path)
    print('rotate '+ str(rotate)+ ' compeleted for '+ postfix+'.jpg')

def noise_adding(img,img_width, img_height,label_name,noise):
    postfix = label_name.split('.')[0]
    output_path = postfix + "_noise_" + str(noise) + ".jpg"
    label_output_path = postfix + '_noise_' + str(noise) + '.xml'

    shutil.copy(label_name,label_output_path)
    add_noise = np.random.randint(noise,size=img.shape,dtype='uint8')
    print(img.shape)

    for i in range(img_height):
        for j in range(img_width):
            for k in range(img.shape[2]):
                if(img[i][j][k] != 255):
                    img[i][j][k] = int(img[i][j][k])+int(add_noise[i][j][k])
                    if img[i][j][k]> 255:
                        img[i][j][k] = 255
    cv2.imencode('.jpg', img)[1].tofile(output_path)
    print('add noise '+ str(noise)+ ' compeleted for '+ postfix+'.jpg')

def add_white(img,img_width, img_height,label_name,scale):
    postfix = label_name.split('.')[0]
    output_path = postfix + "_white" + ".jpg"
    label_output_path = postfix + '_white' + '.xml'

    shutil.copy(label_name, label_output_path)
    tree = ET.parse(label_output_path)
    root = tree.getroot()
    for object in root.iter('object'):
        xmin = int(object.find('bndbox/xmin').text)
        ymin = int(object.find('bndbox/ymin').text)
        xmax = int(object.find('bndbox/xmax').text)
        ymax = int(object.find('bndbox/ymax').text)
        xmin_new = xmin + scale
        ymin_new = ymin + scale
        xmax_new = xmax + scale
        ymax_new = ymax + scale

        object.find('bndbox/xmin').text = str(xmin_new)
        object.find('bndbox/xmax').text = str(xmax_new)
        object.find('bndbox/ymin').text = str(ymin_new)
        object.find('bndbox/ymax').text = str(ymax_new)

    root.find('size/width').text = str(img_width+2*scale)
    root.find('size/height').text = str(img_height+2*scale)
    tree.write(label_output_path, encoding="utf-8")

    new_height = img_height+2*scale
    new_width = img_width + 2*scale
    white_img = np.ones((new_height,new_width,img.shape[2]),dtype=np.uint8)*255

    white_img[scale:new_height-scale,scale:new_width-scale] = img
    cv2.imencode('.jpg', white_img)[1].tofile(output_path)
    print('add white compeleted for '+ postfix+'.jpg' )

def data_augment(IMAGE_PATH,file):
    os.chdir(IMAGE_PATH)
    pos = file.rfind('.')
    postfix = file[:pos] + ".xml"

    file_name = file
    label_name = postfix
    img = cv2.imdecode(np.fromfile(file_name,dtype=np.uint8),-1)#cv2.imread(file_name)
    img_width = img.shape[1]
    img_height = img.shape[0]
    print('origin img_shape:'+str(img.shape))

    #1.镜像mirror
    #create_mirror(label_name, img, img_width, img_height)

    #2.随机裁剪
    #random_ratio = 0.9
    #random_cut(img, random_ratio, label_name, img_width, img_height)

    #3.旋转
    #angle = 90
    #random_rotate(img, label_name, angle, img_width, img_height)

    #4.加噪声
    #noise = 10
    #noise_adding(img, img_width, img_height, label_name, noise)

    #5.补白边
    scale = 400
    add_white(img, img_width, img_height, label_name,scale)


def main():
    class_list = ['钟表类']#['球类','杯子类','书本类','支装物品类']#['塑封包装类','彩妆类','交通工具类','玩具类','肉类']
    path = 'F:\\目标检测\\垃圾分类图片'#sys.argv[1]
    for classname in class_list:
        target_path = os.path.join(path,classname)
        for root,dirs,files in os.walk(target_path,topdown=False):
            IMAGE_PATH = root
            print('img_path:'+root)
            for file in files:
                #if file != '280193_shop_0.jpg':
                #    continue
                if not re.search('.*jpg',file):
                    continue
                if re.search('.*rotate.*',file):
                    continue
                if re.search('.*mirror.*',file):
                    continue
                if re.search('.*bdex.*',file):
                    continue
                #if re.search('.*cut.*',file):
                #    continue
                if re.search('.*white',file):
                    continue
                data_augment(IMAGE_PATH,file)
    '''f = open(sys.argv[1],"r")
    line = f.readline()
    while line:
        line = line.strip('\r\n')#line为txt文件，记录了所有jpg格式
        data_augment(line)
        line = f.readline()
    f.close()'''

if __name__ == '__main__':
    main()
