import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
import cv2
import numpy as np
from copy import deepcopy


##
##image:二值图像
##threshold_point:符合面积条件大小的阈值
def remove_small_points(img):
    img_label = measure.label(img)  # 输出二值图像中所有的连通域
    props = measure.regionprops(img_label)  # 输出连通域的属性，包括面积等

    resMatrix = np.zeros(img_label.shape)
    threshold_point = 0
    new = []
    for i in range(len(props)):
        new.append(props[i].area)
    new.sort(reverse=True)
    # print(new)
    for i in range(len(new)-1):
        if new[i+1] < new[i] * 0.5:
            threshold_point = new[i+1]
    # print(threshold_point)
    for i in range(len(props)):
        if props[i].area > threshold_point:
            tmp = (img_label == i + 1).astype(np.uint8)
            resMatrix += tmp  # 组合所有符合条件的连通域
    resMatrix *= 255
    return resMatrix


def agns(path):
    img = cv2.imread(path, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_gray = cv2.medianBlur(img_gray, 9)
    """same above"""

    block_size = 25
    C = 5

    # ret,th1 = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
    # th2 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,block_size,C)
    th3 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,block_size,C)
    kernel = np.ones((1, 1), np.uint8)
    # th4 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
    th5 = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
    for i in range(len(th5)):
        for j in range(len(th5[0])):
            if th5[i][j] == 0:
                th5[i][j] = 225
            else:
                th5[i][j] =0
    th6 = remove_small_points(th5)
    # titles = ['Original Image', 'Global Thresholding (v = 127)',
    #             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding',
    #           'AM-fs','AG-fs','AG-ns']
    return th6
    #plt.show()