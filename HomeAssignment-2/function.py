import numpy as np
import matplotlib.pyplot as plt
import cv2


def normalization(f):
    f = f / f.max()
    return f


def dft2D(f):
    try:  # f 是一个灰度源图像
        rows, cols = f.shape
        g = np.zeros([rows, cols], dtype=complex)
        for i in range(rows):  # 先对行做一次FFT
            g[i, :] = np.fft.fft(f[i, :])
        for j in range(cols):  # 再对列做一次FFT
            g[:, j] = np.fft.fft(g[:, j])
        return g
    except ValueError:
        print("Input Error!")


def idft2D(g):
    try:  # 其中 g 是一个灰度图像的傅里叶变换
        rows, cols = g.shape
        f = dft2D(g.conjugate())  # 对g的共轭求傅里叶变化
        f = f.conjugate() / (rows * cols)  # 然后再对结果求共轭并除mn即rows*cols
        return f
    except ValueError:
        print("Input Error!")


def fft_slog(g):
    S = np.log(1 + abs(g))
    return S


def center_fft(f):
    m, n = f.shape
    f_shift = normalization(f)
    for i in range(m):
        for j in range(n):
            f_shift[i, j] = f_shift[i, j] * np.power(-1, i + j)
    g = dft2D(f_shift)
    return g


def CheckIfPowerOfTwoAndPadding(img):
    h = img.shape[0]
    w = img.shape[1]
    h_new = 0
    w_new = 0
    if not (h & (h - 1) == 0) and h != 0:
        h_new = np.power(2, int(np.log2(h) + 1))
    else:
        h_new = h
    if not (w & (w - 1) == 0) and w != 0:
        w_new = np.power(2, int(np.log2(w) + 1))
    else:
        w_new = w

    dif_h = int((h_new - h)/2)
    dif_w = int((w_new - w)/2)
    new_img = np.zeros((h_new, w_new))
    new_img[dif_h:dif_h+h, dif_w:dif_w+w] = img.copy().astype(np.int)

    return new_img

