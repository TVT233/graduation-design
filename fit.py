import matplotlib.pyplot as plt
import numpy as np
import cv2
import pylab as mpl

image = cv2.imread('C:/Users/13291/Downloads/yolov5-master/process/result/6/otsu_result/00333_1.jpg')
image = cv2.cvtCOLOR(image,cv2.COLOR_BGR2GRAY)
print(image)
ti = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
yi = [33.40, 79.50, 122.65, 159.05, 189.15, 214.15, 238.65, 252.2, 267.55, 280.50, 296.65, 301.65, 310.4, 318.15,
      325.15]

z1 = np.polyfit(ti, yi, 5)
print(z1)