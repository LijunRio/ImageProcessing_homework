import numpy as np
import matplotlib.pyplot as plt
import cv2
from function import *

img_array = np.zeros((512, 512), dtype=complex)
img_array[256-30:256+30, 256-5:256+5] = 1
# img = Image.fromarray(img_array.real)

img_nor = normalization(img_array)
img_dft_center = center_fft(img_nor)
img_center_log = fft_slog(img_dft_center)
plt.imshow(img_center_log, cmap='gray')
plt.show()
