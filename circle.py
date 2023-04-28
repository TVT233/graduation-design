import cv2
import math
import random
import numpy as np
from numpy.ma import cos, sin
import matplotlib.pyplot as plt
import time

first = 0.001

def get_width(img_origin):
    time1 = time.time()
    img_origin = cv2.cvtColor(img_origin.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    thresh, img=cv2.threshold(img_origin, 128, 255, cv2.THRESH_BINARY)
    # 对输入的图像作阈值分割
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    contous, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contous))
    # 寻找裂缝的轮廓
    circle_list = []
    # 存储各个裂缝部分最大内切圆半径和圆心
    c = contous[0]
    # for c in contous:
    # 定义能包含此裂缝的最小矩形，矩形为水平方向
    # print(c.shape)
    left_x = min(c[:, 0, 0])
    right_x = max(c[:, 0, 0])
    down_y = max(c[:, 0, 1])
    up_y = min(c[:, 0, 1])
    # 最小矩形中最小的边除2，裂缝内切圆的半径最大不超过此距离
    upper_r = min(right_x - left_x, down_y - up_y) / 2
    # 定义相切二分精度precision
    precision = math.sqrt((right_x - left_x) ** 2 + (down_y - up_y) ** 2) / (2 ** 13)
    # 构造包含轮廓的矩形的所有像素点
    Nx = 2 ** 9
    Ny = 2 ** 9
    pixel_X = np.linspace(left_x, right_x, Nx)
    # 在left_x至right_x之间创建了256个均匀分布的间隔点
    pixel_Y = np.linspace(up_y, down_y, Ny)

    # 从坐标向量中生成网格点坐标矩阵
    xx, yy = np.meshgrid(pixel_X, pixel_Y)
    # 筛选出轮廓内所有像素点
    in_list = []
    dist = []
    for i in range(pixel_X.shape[0]):
        for j in range(pixel_X.shape[0]):
            # cv2.pointPolygonTest可查找图像中的点与轮廓之间的最短距离.当点在轮廓外时返回负值，当点在内部时返回正值，如果点在轮廓上则返回零
            # 统计裂缝内的所有点的坐标
            distance = cv2.pointPolygonTest(c, (xx[i][j], yy[i][j]), True)
            if distance > 0:
                # print('此内点到轮廓的最短距离为{}'.format(distance))
                dist.append((xx[i][j], yy[i][j], distance))
    # min = 0
    time2 = time.time()
    for i in range(len(dist)):
        if i > int(first*len(dist)):
            break
        for j in range(len(dist)-i-1):
            if dist[i][2] < dist[i+j+1][2]:
                tmp = dist[i]
                dist[i] = dist[i+j+1]
                dist[i+j+1] = tmp
    for num in range(int(first*len(dist))):
        # in_list.append((xx[i][j], yy[i][j]))
        in_list.append((dist[num][0], dist[num][1]))
    time3 = time.time()
    print('In_list长度为{}'.format(len(in_list)))
    in_point = np.array(in_list)
    # 随机搜索百分之一的像素点提高内切圆半径下限
    N = len(in_point)
    rand_index = random.sample(range(N), N // 100)
    rand_index.sort()
    radius = 0
    big_r = upper_r  # 裂缝内切圆的半径最大不超过此距离
    center = None
    for id in rand_index:
        tr = iterated_optimal_incircle_radius_get(c, in_point[id][0], in_point[id][1], radius, big_r, precision)
        if tr > radius:
            radius = tr
            center = (in_point[id][0], in_point[id][1])  # 只有半径变大才允许位置变更，否则保持之前位置不变
    # 循环搜索剩余像素对应内切圆半径
    time4 = time.time()
    loops_index = [i for i in range(N) if i not in rand_index]
    for id in loops_index:
        tr = iterated_optimal_incircle_radius_get(c, in_point[id][0], in_point[id][1], radius, big_r, precision)
        if tr > radius:
            radius = tr
            center = (in_point[id][0], in_point[id][1])  # 只有半径变大才允许位置变更，否则保持之前位置不变
    time5 = time.time()
    circle_list.append([radius, center])  # 保存每条裂缝最大内切圆的半径和圆心
    # 输出裂缝的最大宽度
    print('裂缝宽度：', round(radius * 2, 2))
    # print('---------------')
    expansion_circle_radius_list = [i[0] for i in circle_list]  # 每条裂缝最大内切圆半径列表
    max_radius = max(expansion_circle_radius_list)
    max_center = circle_list[expansion_circle_radius_list.index(max_radius)][1]
    # print('最大宽度：', round(max_radius * 2, 2))

    # 绘制轮廓
    cv2.drawContours(img_origin, contous, -1, (255, 255, 255), -1)
    # 绘制裂缝轮廓最大内切圆
    for expansion_circle in circle_list:
        radius_s = expansion_circle[0]
        center_s = expansion_circle[1]
        if radius_s == max_radius:  # 最大内切圆，用蓝色标注
            cv2.circle(img_origin, (int(max_center[0]), int(max_center[1])), int(max_radius), (255, 0, 0), 2)
        # else:  # 其他内切圆，用青色标注
            # cv2.circle(img_origin, (int(center_s[0]), int(center_s[1])), int(radius_s), (255, 245, 0), 2)

    # cv2.imshow('Inscribed_circle', img_origin)
    # cv2.imwrite(r'E:\本科毕设\EXP/Inscribed_circle.png', img_origin)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    time6 = time.time()
    print('找内点用时{}'.format(time2 - time1))
    print('排序提取前first部分用时{}'.format(time3 - time2))
    print('前1%内切圆检测用时{}'.format(time4 - time3))
    print('剩余内切圆用时{}'.format(time5 - time4))
    print('绘制用时{}'.format(time6 - time5))
    return img_origin, round(max_radius * 2, 2)



def iterated_optimal_incircle_radius_get(contous, pixelx, pixely, small_r, big_r, precision):
    '''
    计算轮廓内最大内切圆的半径
    Args:
        contous: 轮廓像素点array数组
        pixelx: 圆心x像素坐标
        pixely: 圆心y像素坐标
        small_r: 之前所有计算所求得的内切圆的最大半径，作为下次计算时的最小半径输入，只有半径变大时才允许位置变更，否则保持之前位置不变
        big_r: 圆的半径最大不超过此距离
        precision: 相切二分精度，采用二分法寻找最大半径

    Returns: 轮廓内切圆的半径
    '''
    radius = small_r
    L = np.linspace(0, 2 * math.pi, 360)  # 确定圆散点剖分数360, 720
    circle_X = pixelx + radius * cos(L)
    circle_Y = pixely + radius * sin(L)
    for i in range(len(circle_Y)):
        if cv2.pointPolygonTest(contous, (circle_X[i], circle_Y[i]), False) < 0:  # 如果圆散集有在轮廓之外的点
            return 0
    while big_r - small_r >= precision:  # 二分法寻找最大半径
        half_r = (small_r + big_r) / 2
        circle_X = pixelx + half_r * cos(L)
        circle_Y = pixely + half_r * sin(L)
        if_out = False
        for i in range(len(circle_Y)):
            if cv2.pointPolygonTest(contous, (circle_X[i], circle_Y[i]), False) < 0:  # 如果圆散集有在轮廓之外的点
                big_r = half_r
                if_out = True
        if not if_out:
            small_r = half_r
    radius = small_r
    return radius

if __name__ == '__main__':
    start_time = time.time()
    img_origin = cv2.imread(r'D:\game\source2\source\2.jpg', cv2.IMREAD_GRAYSCALE)
    img, radium = get_width(img_origin)
    end_time = time.time()
    total_time = end_time - start_time
    print('本次计算宽度用时{}'.format(total_time))
    plt.subplot(121)
    plt.imshow(cv2.imread(r'D:\game\source2\source\2.jpg'))
    plt.subplot(122)
    plt.imshow(img)
    plt.show()