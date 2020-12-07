import numpy as np
import matplotlib.pyplot as plt
import cv2
from function import *

img = cv2.imread('./img/rose512.tif', 0)
img_array = np.array(img)
img_norm = normalization(img_array)

# 使用傅里叶变化和中心傅里叶变化两种情况
img_dft = dft2D(img_norm)
img_center_dft = center_fft(img_norm)

# 使用slog函数更好的可视化
img_dft_slog = fft_slog(img_dft)
img_center_slog =  fft_slog(img_center_dft)

print("dft2d:")
print(img_dft.real.max(), img_dft.real.min())
print(img_dft_slog.max(), img_dft_slog.min())
print("fft_centering:")
print(img_center_dft.real.max(), img_center_dft.real.min())
print(img_center_slog.max(), img_center_slog.min())

plt.imshow(img_dft_slog, cmap='gray')
plt.imshow(img_center_slog, cmap='gray')
plt.show()
