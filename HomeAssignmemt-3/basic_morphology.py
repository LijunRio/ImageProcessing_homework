import numpy as np
import cv2


# 腐蚀
def erode(_img):  # 输入的为二值图
    # 初始化图像平移矩阵
    m_1 = np.array([[1, 0, -1], [0, 1, -1]], dtype=np.float32)  # 左上
    m_2 = np.array([[1, 0, 0], [0, 1, -1]], dtype=np.float32)  # 上
    m_3 = np.array([[1, 0, 1], [0, 1, -1]], dtype=np.float32)  # 右上
    m_4 = np.array([[1, 0, -1], [0, 1, 0]], dtype=np.float32)  # 左
    m_5 = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)  # 右
    m_6 = np.array([[1, 0, -1], [0, 1, 1]], dtype=np.float32)  # 左下
    m_7 = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.float32)  # 下
    m_8 = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32)  # 右下
    M = [m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8]

    # 9个平移后的图像取交集得到腐蚀结果
    img_result = _img.copy()
    for i in M:
        img_shift = cv2.warpAffine(_img, i, (_img.shape[1], _img.shape[0]))  # 使用放射变化进行平移
        img_result = cv2.bitwise_and(img_result, img_shift)

    return img_result


# 膨胀
def dilate(_img):  # 输入的为二值图
    # 初始化图像平移矩阵
    m_1 = np.array([[1, 0, -1], [0, 1, -1]], dtype=np.float32)  # 左上
    m_2 = np.array([[1, 0, 0], [0, 1, -1]], dtype=np.float32)  # 上
    m_3 = np.array([[1, 0, 1], [0, 1, -1]], dtype=np.float32)  # 右上
    m_4 = np.array([[1, 0, -1], [0, 1, 0]], dtype=np.float32)  # 左
    m_5 = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)  # 右
    m_6 = np.array([[1, 0, -1], [0, 1, 1]], dtype=np.float32)  # 左下
    m_7 = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.float32)  # 下
    m_8 = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32)  # 右下
    M = [m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8]

    # 9个平移后的图像取交集得到腐蚀结果
    img_result = _img.copy()
    for i in M:
        img_shift = cv2.warpAffine(_img, i, (_img.shape[1], _img.shape[0]))  # 使用放射变化进行平移
        img_result = cv2.bitwise_or(img_result, img_shift)

    return img_result


# 开运算
def open_operation(_img):
    # 先腐蚀, 再膨胀
    img_result = erode(_img)
    img_result = dilate(img_result)

    return img_result
