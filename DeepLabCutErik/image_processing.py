import os
import cv2
import matplotlib.pyplot as plt
import shutil
from distutils.dir_util import copy_tree

from pyrsistent import m




orig_real_path = 'D:\\Chenqi\\KP Detection\\dataset\\sim02'
mod_out_path = 'D:\\Chenqi\\KP Detection\\dataset\\sim02grayinverted'


if not os.path.exists(mod_out_path):
    shutil.copytree(orig_real_path, mod_out_path, ignore=shutil.ignore_patterns('TrainingOutput'))
    #os.makedirs(mod_out_path)
    #os.makedirs(os.path.join(mod_out_path, 'RGB'))
    #copy_tree(orig_real_path, mod_out_path)


for folder in os.listdir(orig_real_path):
    if folder[:3]=='RGB':
        picfolder = folder
files = os.listdir(orig_real_path+f'\\{picfolder}')
print(files)
for file in files:
    image = cv2.imread(os.path.join(orig_real_path,picfolder,file))    
    
    # change contrast
    # hsvImg = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    # hsvImg[...,1] = hsvImg[...,1]*1     #multiple by a factor to change the saturation
    # # hsvImg[...,2] = hsvImg[...,2]*1     #multiple by a factor of less than 1 to reduce the brightness 
    # image=cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)

    # change to gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.merge([image,image,image])
    image = cv2.bitwise_not(image)
    os.remove(os.path.join(mod_out_path,picfolder,file))
    cv2.imwrite(os.path.join(mod_out_path,picfolder,file),image)
    # cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('gray',(224,224))
    # cv2.imshow('gray',image)
    # cv2.waitKey()



# plt.imshow(image)
# plt.show()