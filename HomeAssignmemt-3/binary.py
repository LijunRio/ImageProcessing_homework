import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# 根据阈值分割的迭代法计算阈值T
def iteration_threshold(img_array):
    # 使用图像的均值作为初始阈值
    T = 0
    T_new = np.mean(img_array)
    epsilon = 1e-5
    k = 0

    while np.abs(T_new - T) >= epsilon:
        T = T_new
        G1 = []  # 灰度值大于T的像素组成G1
        G2 = []  # 灰度值小于T的像素组成G2
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                if img_array[i, j] > T:
                    G1.append(img_array[i, j])
                else:
                    G2.append(img_array[i, j])
        T_new = 0.5 * (np.mean(G1) + np.mean(G2))
        k += 1
    return T_new, k


def Binaryzation(_img):
    # 迭代法求阈值
    _threshold, _k = iteration_threshold(_img)
    # 二值化
    img_bin = np.where(_img > _threshold, 0, 255)
    return _threshold, _k, img_bin.astype(np.uint8)

