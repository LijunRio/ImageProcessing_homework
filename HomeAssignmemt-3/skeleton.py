import numpy as np
import cv2
from basic_morphology import open_operation
from basic_morphology import dilate
from basic_morphology import erode


# 基于形态学的骨架提取
def mor_skeleton(_img):  # _img: 待提取骨架图像(默认为前景为白色的二值图像)
    # 骨架图像初始化
    img_result = np.zeros_like(_img)

    # 循环提取骨架, 当腐蚀后图像无前景时停止
    while np.sum(_img):
        # 当第一次的时候k=0， 所以相当于直接开运算
        # 迭代过一次后，_img是一个已经腐蚀并做差后的图像
        img_open = open_operation(_img)
        img_sub = _img - img_open  # 求差
        img_result = cv2.bitwise_or(img_result, img_sub.copy())  # 求并生成骨架
        _img = erode(_img)  # 再进行腐蚀
    return img_result


# 获得8邻域内极大值像素组成的图像
def find_max(_img):
    # 生成8个减法模板
    kmax_1 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    kmax_2 = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    kmax_3 = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    kmax_4 = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float32)
    kmax_5 = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=np.float32)
    kmax_6 = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
    kmax_7 = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float32)
    kmax_8 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
    kernel = [kmax_1, kmax_2, kmax_3, kmax_4, kmax_5, kmax_6,
              kmax_7, kmax_8]

    # 生成一个与原图大小相等像素全为255的图像
    img_result = cv2.bitwise_not(np.zeros_like(_img, dtype=np.uint8))
    # 依次进行减法模板操作, 取结果交集为极大值像素图像
    for i in kernel:
        # 减法模板滤波
        img_m = cv2.filter2D(_img, -1, i)
        # 差值非负处取为255: 操作点像素值>=被减处像素
        img_m = np.where(img_m >= 0.0, 255, 0)
        img_m = img_m.astype(np.uint8)
        # 大于等于8邻域内所有像素的点为区域极大值点
        img_result = cv2.bitwise_and(img_result, img_m)

    return img_result


# 基于距离变化的骨架提取
def distTrans_skeleton(_img):  # img: 待提取骨架图像(默认为前景为白色的二值图像)

    # 通过形态学操作获得前景边界
    img_bd = _img - erode(_img)
    # 对边界图像做距离变换
    img_distTrans = cv2.distanceTransform(
        cv2.bitwise_not(img_bd.copy()),
        cv2.DIST_L2, cv2.DIST_MASK_3)
    # 求距离变换图中的局部极大值
    img_max = find_max(img_distTrans)
    # 落入原二值图像中的局部极大值即为图像的骨架
    img_result = cv2.bitwise_and(img_max, _img)

    return img_result

