from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def scanLine4e(f, I, loc):
    try:
        if loc == 'row':
            s = f[I, :]
        elif loc == 'column':
            s = f[:, I]
        return s
    except ValueError:
        print("Input Error!")


img = Image.open('einstein.tif')
img_arr = np.array(img)
print(img_arr.shape)
img_row = img_arr.shape[0]
img_col = img_arr.shape[1]
sr = scanLine4e(img_arr, round(img_row / 2), 'row')
sc = scanLine4e(img_arr, round(img_col / 2), 'column')
plt.figure('Draw')
plt.plot(sr)
plt.plot(sc)
plt.show()


