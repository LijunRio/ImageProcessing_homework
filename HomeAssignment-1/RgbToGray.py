from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def rgb1gray(f, method):
    try:
        f_array = np.array(f)
        R = f_array[:, :, 0]
        G = f_array[:, :, 1]
        B = f_array[:, :, 2]
        if method == 'average':
            g = R / 3 + G / 3 + B / 3
        elif method == 'NTSC':
            g = R * 0.2989 + G * 0.5870 + B * 0.1140
        return g
    except ValueError:
        print("Input Error!")

