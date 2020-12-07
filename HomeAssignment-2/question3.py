import numpy as np
import matplotlib.pyplot as plt
import cv2
from function import *

img = cv2.imread('./img/rose512.tif', 0)
img_array = np.array(img)
img_norm = normalization(img_array)

img_dft = dft2D(img_norm)
img_dft_idft = idft2D(img_dft)
diff = img_norm - img_dft_idft.real

plt.imshow(diff, cmap='gray')
plt.show()