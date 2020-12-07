import numpy as np
import cv2
from basic_morphology import erode
from basic_morphology import dilate
from basic_morphology import open_operation


def thinning_algorithm(img, K):  # img: 待细化图像  K: 结构子序列
    img_result = img / 255  # 归一
    # 初始化用于保存上一次结果的矩阵
    img_old = 1 - img_result

    # 循环细化.直至图像保持不变
    while np.sum(img_result - img_old):
        img_old = img_result
        for i in K:
            # 基于卷积结果的击中击不中
            img_temp = np.where(cv2.filter2D(img_result.copy(), -1, i
                                             , borderType=0) == 15, 1, 0)
            img_result = img_result - img_temp

    img_result *= 255
    return img_result.astype(np.uint8)


def thinning(_img):  # img: 待细化图像

    # 生成8个结构子序列
    k_1 = np.array([[16, 16, 16], [0, 1, 0], [2, 4, 8]], dtype=np.uint8)
    k_2 = np.array([[0, 16, 16], [1, 2, 16], [4, 8, 0]], dtype=np.uint8)
    k_3 = np.array([[1, 0, 16], [2, 4, 16], [8, 0, 16]], dtype=np.uint8)
    k_4 = np.array([[1, 2, 0], [4, 8, 16], [0, 16, 16]], dtype=np.uint8)
    k_5 = np.array([[1, 2, 4], [0, 8, 0], [16, 16, 16]], dtype=np.uint8)
    k_6 = np.array([[0, 1, 2], [16, 4, 8], [16, 16, 0]], dtype=np.uint8)
    k_7 = np.array([[16, 0, 1], [16, 2, 4], [16, 0, 8]], dtype=np.uint8)
    k_8 = np.array([[16, 16, 0], [16, 1, 2], [0, 4, 8]], dtype=np.uint8)

    K = [k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8]

    # 细化操作
    img_result = thinning_algorithm(_img, K)

    return img_result


#  找到端节点
def find_end(_img, K):  # _img: 输入图像 K: 结构子序列
    # 像素归一化
    img_ones = _img / 255
    img_result = np.zeros_like(_img, dtype=np.uint8)

    # 利用结构子序列寻找端点
    for i in K:
        img_temp = np.where(cv2.filter2D(img_ones.copy(), -1, i,
                                         borderType=0) == 3, 1, 0)
        img_result = img_result + img_temp

    img_result *= 255
    # 返回只有端节点为前景的图像
    return img_result.astype(np.uint8)


def tailor(_img):  # _img: 待裁剪图像

    # 生成8个结构子
    k_1 = np.array([[0, 4, 4], [1, 2, 4], [0, 4, 4]], dtype=np.uint8)
    k_2 = np.array([[0, 1, 0], [4, 2, 4], [4, 4, 4]], dtype=np.uint8)
    k_3 = np.array([[4, 4, 0], [4, 1, 2], [4, 4, 0]], dtype=np.uint8)
    k_4 = np.array([[4, 4, 4], [4, 1, 4], [0, 2, 0]], dtype=np.uint8)
    k_5 = np.array([[1, 4, 4], [4, 2, 4], [4, 4, 4]], dtype=np.uint8)
    k_6 = np.array([[4, 4, 1], [4, 2, 4], [4, 4, 4]], dtype=np.uint8)
    k_7 = np.array([[4, 4, 4], [4, 1, 4], [4, 4, 2]], dtype=np.uint8)
    k_8 = np.array([[4, 4, 4], [4, 1, 4], [2, 4, 4]], dtype=np.uint8)

    K = [k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8]

    # 细化(去除3个像素组成的分支)
    img_thin = thinning_algorithm(_img, K)
    # 找端点
    img_end = find_end(img_thin, K)
    # 膨胀运算,捡回误伤元素
    img_dilate = img_end
    for _ in range(3):
        img_dilate = dilate(img_dilate)
        img_dilate = cv2.bitwise_and(img_dilate, _img)
    # 获得裁剪结果
    img_result = cv2.bitwise_or(img_dilate, img_thin)

    return img_result
