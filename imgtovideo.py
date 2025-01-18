# import cv2
# import os

# size = (768, 256)

# videowrite = cv2.VideoWriter(r"E:/new_workplace_2023_for_NewNet3", -1, 30, size)
# img_array = []

# path = r"E:/new_workplace_2023_for_NewNet3/tmp"

# for fn in os.listdir(path):
#     filename = os.path.join(path, fn)
#     img = cv2.imread(filename)
#     if img is None:
#         print(filename + "is null")
#         continue
#     videowrite.write(img)

# videowrite.release()
# print('end!')

# -*- coding: UTF-8 -*-
# '''
# @author: mengting gu
# @contact: 1065504814@qq.com
# @time: 2020/12/29 14:14
# @file: jpg2avi.py
# @desc: 
# '''
 
# import os
# import cv2
# import platform
 
# sysstr = platform.system()
# if (sysstr == "Windows"):
#     path = "E:/new_workplace_2023_for_NewNet3/tmp" # '需要调用的图片路径 例如：C:/picture/'
# else:
#     path = "E:/new_workplace_2023_for_NewNet3/tmp"
# filelist = os.listdir(path)
 
# fps = 8 #视频每秒24帧
# size = (768, 256) #需要转为视频的图片的尺寸
# #可以使用cv2.resize()进行修改
 
# video = cv2.VideoWriter("VideoTest1.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
# #视频保存在当前目录下
 
# for item in filelist:
#     if item.endswith('.jpg'):
#         print("item : {}".format(item))
#     #找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
#         item = path + item
#         img = cv2.imread(item)
#         img = cv2.resize(img, (768, 256))
#         # cv2.imshow('img', img)
#         # cv2.waitKey(int(1000/int(fps)))
#         video.write(img)
 
# video.release()
 
# if (sysstr == "Windows"):
#     cv2.destroyAllWindows()
 
 # coding=utf-8
import os
import cv2
from PIL import Image
 
def makevideo(path, fps):
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    path1 = 'E:/new_workplace_2023_for_NewNet3/tmp'
    im = Image.open('E:/new_workplace_2023_for_NewNet3/tmp/0.jpg')
    print(im.size)
    vw = cv2.VideoWriter(path, fourcc, fps, im.size)
    for i in os.listdir(path1):
        frame = cv2.imread(path1 +'/'+ i)
        vw.write(frame)
 
if __name__ == '__main__':
    video_path = 'E:/new_workplace_2023_for_NewNet3/test_new1.mp4'
    makevideo(video_path, 10)  # 图片转视频