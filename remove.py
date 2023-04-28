#coding=utf-8
import os
import shutil

old_path = r'C:\Users\13291\Downloads\yolov5-master\process\detect_result'  # 要复制的文件所在目录
new_path = r'C:\Users\13291\Downloads\yolov5-master\process\else'  #新路径

def remove(path, path_new):
    for ipath in os.listdir(path):
        fulldir = os.path.join(path, ipath)  # 拼接成绝对路径
        #print(fulldir)         #打印相关后缀的文件路径及名称
        shutil.copy(fulldir,path_new)



