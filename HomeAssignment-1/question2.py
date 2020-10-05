from RgbToGray import rgb1gray
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img1 = Image.open('mandril_color.tif')
img2 = Image.open('lena512color.tiff')

imGray1_average = rgb1gray(img1, 'average')
imGray1_NTSC = rgb1gray(img1, 'NTSC')

imGray2_average = rgb1gray(img2, 'average')
imGray2_NTSC = rgb1gray(img2, 'NTSC')

plt.subplot(2, 3, 1)
plt.title('mandril原图')
plt.imshow(img1)

plt.subplot(2, 3, 2)
plt.title('mandril的average灰度')
plt.imshow(imGray1_average, 'gray')

plt.subplot(2, 3, 3)
plt.title('mandril的NTSC灰度')
plt.imshow(imGray1_NTSC, 'gray')

plt.subplot(2, 3, 4)
plt.title('lena原图')
plt.imshow(img2)

plt.subplot(2, 3, 5)
plt.title('lena的average灰度')
plt.imshow(imGray2_average, 'gray')

plt.subplot(2, 3, 6)
plt.title('lena的NTSC灰度')
plt.imshow(imGray2_NTSC, 'gray')

plt.show()
