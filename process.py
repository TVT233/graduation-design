import detect_LocOP
import cut
import Otsu
import getfiles
import remove
import numpy as np
import cv2
import findresultnum
import os
import gaussian
import length


# YOLO的路径得去detect_LocOP.py里改
import circle

FileRoot_buffer = r'C:/Users/13291/Downloads/yolov5-master/process/buffer' # 缓冲目录
FileRoot_source = r'C:/Users/13291/Downloads/yolov5-master/process/buffer/source' # 需裁剪文件根目录
FileRoot_detectResult = r'C:\Users\13291\Downloads\yolov5-master\process\buffer\detect_result' # 检测图像目录
FileRoot_locations = r'C:/Users/13291/Downloads/yolov5-master/process/buffer/detect_location' # 裁剪标签根目录
FileRoot_cutResult = r'C:/Users/13291/Downloads/yolov5-master/process/buffer/cut_result' # 裁剪后文件保存目录
FileRoot_segResult = r'C:/Users/13291/Downloads/yolov5-master/process/buffer/seg_result' # 分割后文件保存目录
FileRoot_result = r'C:/Users/13291/Downloads/yolov5-master/process/result'
FileRoot_lengthResult = r'C:\Users\13291\Downloads\yolov5-master\process\buffer\length'
FileRoot_widthResult = r'C:\Users\13291\Downloads\yolov5-master\process\buffer\width'
FileRoot_info = r'C:\Users\13291\Downloads\yolov5-master\process\buffer\length_width_info'
FileRoot_vision = r'C:\Users\13291\Downloads\yolov5-master\process\buffer\vision_result'
img_width = 640
img_height = 480

# buffer下所属文件夹
Dir_dic = ['cut_result', 'detect_location', 'detect_result', 'seg_result', 'source', 'length_width_info', 'length',
           'width', 'vision_result']


if __name__ == "__main__":
    # 将buffer/source下的图片进行裂缝识别，结果保存至buffer/detect_result & buffer/detect_location,这个路径得自己去detect.py里改
    opt = detect_LocOP.parse_opt()
    detect_LocOP.main(opt)
    # 裁剪裂缝部分
    files, names = getfiles.getfiles(FileRoot_source)  # 获取需裁剪的所有图片的路径和名称
    locations, locations_names = getfiles.getfiles(FileRoot_locations)  # 获取所有图片对应的坐标
    for num in range(len(files)):  # 此处减1是排除待裁剪图片目录里有个坐标文件夹
        # print(locations[num])
        # print(locations_names[num])
        cut.cut(files[num], locations[num], names[num])  # 裁剪图片
    # Otsu分割并腐蚀
    cut_files, cut_names = getfiles.getfiles(FileRoot_cutResult)
    for num in range(len(cut_files)):
        # otsu_result = Otsu.otsu(files[num])
        # kernel = np.ones((8, 8), np.uint8)
        # res = cv2.morphologyEx(otsu_result, cv2.MORPH_OPEN, kernel)
        gaussian_result = gaussin.agns(cut_files[num])
        # print(gaussin_result.shape)
        gaussian_save_path = FileRoot_segResult + '/' + cut_names[num] + '.jpg'
        length_save_path = FileRoot_lengthResult + '/' + cut_names[num] + '.jpg'
        width_save_path = FileRoot_widthResult + '/' + cut_names[num] + '.jpg'
        info_save_path = FileRoot_info + '/' + cut_names[num] + '.txt'
        circle_result, w = circle.get_width(gaussian_result)
        cv2.imwrite(width_save_path, circle_result)
        cv2.imwrite(gaussian_save_path, gaussian_result)
        length_result, l = length.length(gaussian_result)
        cv2.imwrite(length_save_path, length_result)
        file = open(info_save_path, 'w+')
        file.write('pixel_length:' + str(l))
        file.write('\n')
        file.write('pixel_width:' + str(w))
        file.write('\n')
        file.write('length:' + str(round(l*0.9843, 2)) + 'mm')
        file.write('\n')
        file.write('width:' + str(round(w*0.104, 2)) + 'mm')
        file.close()
    for img_num in range(len(files)):
        loc = open(locations[img_num], 'r')
        loc_lines = loc.readlines()
        detect_img = cv2.imread(FileRoot_detectResult + '/' + names[img_num] + '.jpg')
        for line_num in range(len(loc_lines)):
            info = ''
            w = detect_img.shape[1]
            h = detect_img.shape[0]
            msg = loc_lines[line_num].split(" ")  # 利用空格符将该行坐标分为5部分，分别为 ‘类别 X Y W H’
            x1 = max(0, int((float(msg[1]) - float(msg[3]) / 2) * w))  # x_center - width/2
            y1 = max(0, int((float(msg[2]) - float(msg[4]) / 2) * h))  # y_center - height/2
            x2 = max(0, int((float(msg[1]) + float(msg[3]) / 2) * w))  # x_center + width/2
            y2 = max(0, int((float(msg[2]) + float(msg[4]) / 2) * h))  # y_center + height/2
            lwinfo = open(FileRoot_info + '/' + names[img_num] + '_' + str(line_num + 1) + '.txt')
            info_lines = lwinfo.readlines()
            for i in info_lines:
                cv2.putText(detect_img, i.strip('\n'), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                y1 += 25
            lwinfo.close()
        cv2.imwrite(FileRoot_vision + '/' + names[img_num] + '.jpg', detect_img)
        loc.close()

    # 将检测结果和过程中结果转移到储存文件夹中
    rn = findresultnum.findrn(FileRoot_result)
    os.mkdir(FileRoot_result + '/' + str(rn))
    for i in Dir_dic:
        os.mkdir(FileRoot_result + '/' + str(rn) + '/' + i)
        remove.remove(FileRoot_buffer + '/' + i, FileRoot_result + '/' + str(rn) + '/' + i) #把每个文件从缓存目录复制到存储目录
        removelist = os.listdir(FileRoot_buffer + '/' + i) # 删除缓存目录中所有内容
        for j in removelist:
            os.remove(FileRoot_buffer + '/' + i + '/' + j)


