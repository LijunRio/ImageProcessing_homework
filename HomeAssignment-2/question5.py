import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from function import *

img1 = cv2.imread('./img/house.tif', 0)
img2 = Image.open('./img/house02.tif')
img3 = Image.open('./img/lena_gray_512.tif')
img4 = Image.open('./img/lunar_surface.tif')
img5 = Image.open('./img/Characters_test_pattern.tif')

img1_array = np.array(img1)
img2_array = np.array(img2)
img3_array = np.array(img3)
img4_array = np.array(img4)
img5_array = np.array(img5)

img1_checkpower = CheckIfPowerOfTwoAndPadding(img1_array)
img2_checkpower = CheckIfPowerOfTwoAndPadding(img2_array)
img3_checkpower = CheckIfPowerOfTwoAndPadding(img3_array)
img4_checkpower = CheckIfPowerOfTwoAndPadding(img4_array)
img5_checkpower = CheckIfPowerOfTwoAndPadding(img5_array)
img1_norm = normalization(img1_checkpower)
img2_norm = normalization(img2_checkpower)
img3_norm = normalization(img3_checkpower)
img4_norm = normalization(img4_checkpower)
img5_norm = normalization(img5_checkpower)

print(img1_array.shape, img2_array.shape, img3_array.shape, img4_array.shape, img5_array.shape)
print(img1_norm.shape, img2_norm.shape, img3_norm.shape, img4_norm.shape, img5_norm.shape)

dft1 = fft_slog(center_fft(img1_checkpower))
dft2 = fft_slog(center_fft(img2_checkpower))
dft3 = fft_slog(center_fft(img3_checkpower))
dft4 = fft_slog(center_fft(img4_checkpower))
dft5 = fft_slog(center_fft(img5_checkpower))

plt.subplot(1, 2, 1)
plt.title('origin image')
plt.imshow(img5, 'gray')

plt.subplot(1, 2, 2)
plt.title('fft')
plt.imshow(dft5, 'gray')
plt.show()
