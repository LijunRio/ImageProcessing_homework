from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scanLine import scanLine4e

img1 = Image.open('einstein.tif')
img_arr1 = np.array(img1)
print(img_arr1.shape)
img_row1 = img_arr1.shape[0]
img_col1 = img_arr1.shape[1]
sr1 = scanLine4e(img_arr1, round(img_row1 / 2), 'row')
sc1 = scanLine4e(img_arr1, round(img_col1 / 2), 'column')

img2 = Image.open('cameraman.tif')
img_arr2 = np.array(img2)
print(img_arr2.shape)
img_row2 = img_arr2.shape[0]
img_col2 = img_arr2.shape[1]
sr2 = scanLine4e(img_arr2, round(img_row2 / 2), 'row')
sc2 = scanLine4e(img_arr2, round(img_col2 / 2), 'column')

plt.subplot(2, 1, 1)
plt.xlabel('位置坐标')
plt.ylabel('灰度值')
plt.title('einstein的图像扫描值')
L1, = plt.plot(sr1)
L2, = plt.plot(sc1)
plt.legend(handles=[L1, L2], labels=['中心行元素', '中心列元素'], loc='upper right')

plt.subplot(2, 1, 2)
plt.xlabel('位置坐标')
plt.ylabel('灰度值')
plt.title('cameraman的图像扫描值')
L1, = plt.plot(sr2)
L2, = plt.plot(sc2)
plt.legend(handles=[L1, L2], labels=['中心行元素', '中心列元素'], loc='upper right')

plt.show()
