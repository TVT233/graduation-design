import numpy as np
from skimage import io
from skimage.morphology import medial_axis, skeletonize
from skimage import measure
from skimage import data
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import cv2


def show_2dpoints(pointcluster, s=None, quivers=None, qscale=1):
    # pointcluster should be a list of numpy ndarray
    # This functions would show a list of pint cloud in different colors
    n = len(pointcluster)
    nmax = n
    if quivers is not None:
        nq = len(quivers)
        nmax = max(n, nq)

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'tomato', 'gold']
    if nmax < 10:
        colors = np.array(colors[0:nmax])
    else:
        colors = np.random.rand(nmax, 3)

    fig = plt.figure(num=1)
    ax = fig.add_subplot(1, 1, 1)

    if s is None:
        s = np.ones(n) * 2

    for i in range(n):
        ax.scatter(pointcluster[i][:, 0], pointcluster[i][:, 1], s=s[i], c=[colors[i]], alpha=0.6)

    if quivers is not None:
        for i in range(nq):
            ax.quiver(quivers[i][:, 0], quivers[i][:, 1], quivers[i][:, 2], quivers[i][:, 3], color=[colors[i]],
                      scale=qscale)

    plt.show()


def SVD(points):
    # 二维，三维均适用
    # 二维直线，三维平面
    pts = points.copy()
    # 奇异值分解
    c = np.mean(pts, axis=0)
    A = pts - c  # shift the points
    A = A.T  # 3*n
    u, s, vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)  # A=u*s*vh
    normal = u[:, -1]

    # 法向量归一化
    nlen = np.sqrt(np.dot(normal, normal))
    normal = normal / nlen
    # normal 是主方向的方向向量 与PCA最小特征值对应的特征向量是垂直关系
    # u 每一列是一个方向
    # s 是对应的特征值
    # c >>> 点的中心
    # normal >>> 拟合的方向向量
    return u, s, c, normal


def calcu_dis_from_ctrlpts(ctrlpts):
    if ctrlpts.shape[1] == 4:
        return np.sqrt(np.sum((ctrlpts[:, 0:2] - ctrlpts[:, 2:4]) ** 2, axis=1))
    else:
        return np.sqrt(np.sum((ctrlpts[:, [0, 2]] - ctrlpts[:, [3, 5]]) ** 2, axis=1))


def estimate_normal_for_pos(pos, points, n):
    # estimate normal vectors at a given point
    pts = np.copy(points)
    tree = KDTree(pts, leaf_size=2)
    idx = tree.query(pos, k=n, return_distance=False, dualtree=False, breadth_first=False)
    # pts = np.concatenate((np.concatenate((pts[0].reshape(1,-1),pts),axis=0),pts[-1].reshape(1,-1)),axis=0)
    normals = []
    for i in range(0, pos.shape[0]):
        pts_for_normals = pts[idx[i, :], :]
        _, _, _, normal = SVD(pts_for_normals)
        normals.append(normal)
    normals = np.array(normals)
    return normals


def estimate_normals(points, n):
    pts = np.copy(points)
    tree = KDTree(pts, leaf_size=2)
    idx = tree.query(pts, k=n, return_distance=False, dualtree=False, breadth_first=False)
    # pts = np.concatenate((np.concatenate((pts[0].reshape(1,-1),pts),axis=0),pts[-1].reshape(1,-1)),axis=0)
    normals = []
    for i in range(0, pts.shape[0]):
        pts_for_normals = pts[idx[i, :], :]
        _, _, _, normal = SVD(pts_for_normals)
        normals.append(normal)
    normals = np.array(normals)
    return normals


def get_crack_ctrlpts(centers, normals, bpoints, hband=5, vband=2):
    # main algorithm to obtain crack width
    cpoints = np.copy(centers)
    cnormals = np.copy(normals)

    xmatrix = np.array([[0, 1], [-1, 0]])
    cnormalsx = np.dot(xmatrix, cnormals.T).T  # the normal of x axis
    N = cpoints.shape[0]

    interp_segm = []
    widths = []
    for i in range(N):
        try:
            ny = cnormals[i]
            nx = cnormalsx[i]
            tform = np.array([nx, ny])
            bpoints_loc = np.dot(tform, bpoints.T).T
            cpoints_loc = np.dot(tform, cpoints.T).T
            ci = cpoints_loc[i]

            bl_ind = (bpoints_loc[:, 0] - (ci[0] - hband)) * (bpoints_loc[:, 0] - ci[0]) < 0
            br_ind = (bpoints_loc[:, 0] - ci[0]) * (bpoints_loc[:, 0] - (ci[0] + hband)) <= 0
            bl = bpoints_loc[bl_ind]  # left points
            br = bpoints_loc[br_ind]  # right points

            blt = bl[bl[:, 1] > np.mean(bl[:, 1])]
            if np.ptp(blt[:, 1]) > vband:
                blt = blt[blt[:, 1] > np.mean(blt[:, 1])]

            blb = bl[bl[:, 1] < np.mean(bl[:, 1])]
            if np.ptp(blb[:, 1]) > vband:
                blb = blb[blb[:, 1] < np.mean(blb[:, 1])]

            brt = br[br[:, 1] > np.mean(br[:, 1])]
            if np.ptp(brt[:, 1]) > vband:
                brt = brt[brt[:, 1] > np.mean(brt[:, 1])]

            brb = br[br[:, 1] < np.mean(br[:, 1])]
            if np.ptp(brb[:, 1]) > vband:
                brb = brb[brb[:, 1] < np.mean(brb[:, 1])]

            # bh = np.vstack((bl,br))
            # bmax = np.max(bh[:,1])
            # bmin = np.min(bh[:,1])

            # blt = bl[bl[:,1]>bmax-vband] # left top points
            # blb = bl[bl[:,1]<bmin+vband] # left bottom points

            # brt = br[br[:,1]>bmax-vband] # right top points
            # brb = br[br[:,1]<bmin+vband] # right bottom points

            t1 = blt[np.argsort(blt[:, 0])[-1]]
            t2 = brt[np.argsort(brt[:, 0])[0]]

            b1 = blb[np.argsort(blb[:, 0])[-1]]
            b2 = brb[np.argsort(brb[:, 0])[0]]

            interp1 = (ci[0] - t1[0]) * ((t2[1] - t1[1]) / (t2[0] - t1[0])) + t1[1]
            interp2 = (ci[0] - b1[0]) * ((b2[1] - b1[1]) / (b2[0] - b1[0])) + b1[1]

            if interp1 - ci[1] > 0 and interp2 - ci[1] < 0:
                widths.append([i, interp1 - ci[1], interp2 - ci[1]])

                interps = np.array([[ci[0], interp1], [ci[0], interp2]])

                interps_rec = np.dot(np.linalg.inv(tform), interps.T).T

                # show_2dpoints([bpointsxl_loc1,bpointsxl_loc2,bpointsxr_loc1,bpointsxr_loc2,np.array([ptsl_1,ptsl_2]),np.array([ptsr_1,ptsr_2]),interps,ci.reshape(1,-1)],s=[1,1,1,1,20,20,20,20])
                interps_rec = interps_rec.reshape(1, -1)[0, :]
                interp_segm.append(interps_rec)
        except:
            print("the %d-th was wrong" % i)
            continue
    interp_segm = np.array(interp_segm)
    widths = np.array(widths)
    # check
    # show_2dpoints([np.array([[ci[0],interp1],[ci[0],interp2]]),np.array([t1,t2,b1,b2]),cpoints_loc,bl,br],[10,20,15,2,2])
    return interp_segm, widths

def length(img_bi):
    path = r"C:\Users\13291\Desktop\image\source_AGNS"

    image = img_bi # 以灰度图读入
    # print(image[0])
    iw, ih = image.shape
    # 获取图片宽度和高度

    blobs = np.copy(image) # 图像二值化
    blobs[blobs < 128] = 0
    blobs[blobs > 128] = 1

    blobs = blobs.astype(np.uint8) # 转换为Uint8
    # Generate the data
    # blobs = data.binary_blobs(200, blob_size_fraction=.2,
    # volume_fraction=.35, seed=1)
    # using scikit-image
    ## Compute the medial axis (skeleton) and the distance transform
    # skel, distance = medial_axis(blobs, return_distance=True)
    ## Distance to the background for pixels of the skeleton
    # dist_on_skel = distance * skel

    # Compare with other skeletonization algorithms
    skeleton = skeletonize(blobs)
    # 图形骨架化
    # skeleton_lee = skeletonize(blobs, method='lee')
    x, y = np.where(skeleton > 0) # 返回骨架像素的坐标，分别存放在x, y
    # print(x)
    # print(y)
    centers = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    print('centers are {}'.format(centers))
    # 列向量重新组合，获得所有骨架像素的坐标
    normals = estimate_normals(centers, 3)

    # search contours of the crack
    contours = measure.find_contours(blobs, 0.8)

    bl = contours[0]
    br = contours[1]

    bpoints = np.vstack((bl, br))

    # interp_segm, widths = get_crack_ctrlpts(centers,normals,bpoints,hband=2,vband=2)


    bpixel = np.zeros((iw, ih, 3), dtype=np.uint8)
    bpoints = bpoints.astype(np.int_)
    bpixel[bpoints[:, 0], bpoints[:, 1], 0] = 255

    skeleton_pixel = np.zeros((iw, ih, 3), dtype=np.uint8)
    skeleton_pixel[skeleton, 1] = 255

    bpixel_and_skeleton = np.copy(bpixel)
    bpixel_and_skeleton[skeleton, 1] = 255
    print("裂缝长度：{}".format(len(x)))
    return bpixel_and_skeleton, len(x)

if __name__ == '__main__':
    img = cv2.imread(r'C:\Users\13291\Downloads\yolov5-master\process\result\31\seg_result\1_1.jpg', cv2.IMREAD_GRAYSCALE)
    res, l = length(img)
    plt.imshow(res)
    plt.show()


    # fig, axes = plt.subplots(1, 3, figsize=(8, 8))
    # ax = axes.ravel()
    #
    # ax[0].imshow(blobs, cmap=plt.cm.gray)
    # ax[0].axis('off')
    # cv2.imwrite(path + '/long_1.jpg', bpixel_and_skeleton)
    # ax[1].imshow(bpixel_and_skeleton)
    # # for contour in contours:
    # #    ax[1].plot(contour[:, 1], contour[:, 0], linewidth=2)
    #
    # # for i in range(interp_segm.shape[0]):
    # #    ax[1].plot([interp_segm[i,1],interp_segm[i,3]],[interp_segm[i,0],interp_segm[i,2]],'-b')
    #
    # # ax[1].set_title('medial_axis')
    # ax[1].axis('off')
    #
    # # ================ small window ==================
    # pos = np.array([191, 291]).reshape(1, -1)  # input (x,y) where need to calculate crack width
    # # pos = np.array([142, 178]).reshape(1,-1)
    #
    # posn = estimate_normal_for_pos(pos, centers, 3)
    #
    # interps, widths2 = get_crack_ctrlpts(pos, posn, bpoints, hband=1.5, vband=2)
    #
    # sx = pos[0, 0] - 20
    # sy = pos[0, 1] - 20
    #
    # #ax[2].imshow(bpixel_and_skeleton)
    # print(interps.shape[0])
    # for i in range(interps.shape[0]):
    #     ax[2].plot([interps[i, 1], interps[i, 3]], [interps[i, 0], interps[i, 2]], c='c', ls='-', lw=5, marker='o', ms=8,
    #                mec='c', mfc='c')
    #
    # ax[2].set_ylim(sx, sx + 40)
    # ax[2].set_xlim(sy, sy + 40)
    #
    # # ax[2].set_title('skeletonize')
    # ax[2].axis('off')
    #
    # print(interps)
    #
    # fig.tight_layout()
    #
    # plt.show()


