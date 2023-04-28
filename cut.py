import os
import cv2
import getfiles

FileRoot_cut = r'C:/Users/13291/Downloads/yolov5-master/process/buffer/source' # 需裁剪文件根目录
FileRoot_save_image = r'C:\Users\13291\Desktop\image\imgae_roi'
FileRoot_save_GR = r'C:\Users\13291\Desktop\image\ground_truth_roi'
FileRoot_locations = r'C:/Users/13291/Downloads/yolov5-master/process/buffer/detect_location' # 裁剪标签根目录
FileRoot_GR = r'C:\Users\13291\Desktop\image\ground_truth' # 裁剪GR根目录
FileRoot_save = r'C:/Users/13291/Downloads/yolov5-master/process/buffer/cut_result' # 裁剪后文件保存目录
def cut(image,location,name):
    f = open(location, 'r+', encoding='utf-8') # 打开坐标所在的文件
    lines = f.readlines() # 读取所有坐标值，共几行代表有几个目标
    if len(lines) == 0: # 如果没目标，直接返回原图
        img_roi = cv2.imread(image)
        save_path = FileRoot_save + '/' + name + '_1' + '.jpg'
        cv2.imwrite(save_path, img_roi)
    else:
        for i in range(len(lines)):
            img = cv2.imread(image) # 读取待裁剪的图片
            w = img.shape[1] # 读取图片像素高度
            h = img.shape[0] # 读取图片像素宽度
            msg = lines[i].split(" ") # 利用空格符将该行坐标分为5部分，分别为 ‘类别 X Y W H’
            #print(msg)
            x1 = max(0,int((float(msg[1]) - float(msg[3]) / 2) * w))  # x_center - width/2
            y1 = max(0,int((float(msg[2]) - float(msg[4]) / 2) * h))  # y_center - height/2
            x2 = max(0,int((float(msg[1]) + float(msg[3]) / 2) * w))  # x_center + width/2
            y2 = max(0,int((float(msg[2]) + float(msg[4]) / 2) * h))  # y_center + height/2
            #print(x1,x2,y1,y2)
            img_roi = img[y1:y2, x1:x2] # 利用转换后的坐标进行裁剪
            save_path = FileRoot_save + '/' + name + '_' + str(i + 1) + '.jpg'
            # print(img_roi)
            cv2.imwrite(save_path, img_roi)


if __name__ == '__main__':
    files, names = getfiles.getfiles(FileRoot_cut) # 获取需裁剪的所有图片的路径和名称
    locations, locations_names = getfiles.getfiles(FileRoot_locations)# 获取所有图片对应的坐标
    for num in range(len(files)): #此处减1是排除待裁剪图片目录里有个坐标文件夹
        cut(files[num], locations[num], names[num]) # 裁剪图片



