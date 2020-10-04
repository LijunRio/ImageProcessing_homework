from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def rgb1gray(f, method):
    try:
        R = f[:, :, 0]
        G = f[:, :, 1]
        B = f[:, :, 2]
        if method == 'average':
            g = R / 3 + G / 3 + B / 3
        elif method == 'NTSC':
            g = R * 0.2989 + G * 0.5870 + B * 0.1140
        return g
    except ValueError:
        print("Input Error!")


img = Image.open('mandril_color.tif')
img_arr = np.array(img)
imGray = rgb1gray(img_arr, 'average')
plt.imshow(imGray)
plt.show()
