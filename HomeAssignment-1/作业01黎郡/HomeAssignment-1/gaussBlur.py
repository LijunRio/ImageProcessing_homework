import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from RgbToGray import rgb1gray


def twodConv(f, w, padding):
    try:
        f_array = np.array(f)
        kernel_size = w.shape[0]
        H, W = f_array.shape
        pad = kernel_size // 2
        g = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
        g[pad:pad + H, pad:pad + W] = f.copy().astype(np.float)
        if padding == 'replicate':
            # 左右列补齐
            g[0:pad, pad:-pad] = f_array[0:pad, :]
            g[-pad:, pad:-pad] = f_array[-pad:, :]
            # 上下行补齐
            g[pad:-pad, 0:pad] = f_array[:, 0:pad]
            g[pad:-pad, -pad:] = f_array[:, -pad:]
            # 四角元素补齐
            g[0, 0] = f_array[pad, pad]
            g[-pad, pad] = f_array[-pad, pad]
            g[0, -pad] = f_array[0, -pad]
            g[-pad, -pad] = f_array[-pad, -pad]

        # Convolution process
        tmp = g.copy()
        # rotate 180 degrees
        for y in range(H):
            for x in range(W):
                g[pad + y, pad + x] = np.sum(w * tmp[y:y + kernel_size, x:x + kernel_size])
        out = g[pad:pad + H, pad:pad + W].astype(np.uint8)
        return out

    except ValueError:
        print("Input Error!")


# 中 sig 对应于高斯函数定义中的σ,w
# 的大小为 m×m
def gaussKernel(sig, m):
    # kernel_size = 3sigma*2 + 1
    kernel_size = round(3 * sig * 2 + 1)
    if m < kernel_size:
        print("m is too small !")
    else:
        kernel_size = m
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float)
    pad = kernel_size // 2
    for x in range(-pad, -pad + kernel_size):
        for y in range(-pad, -pad + kernel_size):
            kernel[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sig ** 2)))
    kernel /= (sig * np.sqrt(2 * np.pi))
    kernel /= kernel.sum()
    return kernel


