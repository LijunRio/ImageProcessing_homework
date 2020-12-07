from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def scanLine4e(f, I, loc):
    try:
        if loc == 'row':
            s = f[I, :]
        elif loc == 'column':
            s = f[:, I]
        return s
    except ValueError:
        print("Input Error!")


