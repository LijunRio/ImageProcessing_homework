import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from auxiliary_function import *


img_array = np.zeros((512, 512))
img_array[256 - 30:256 + 30, 256 - 5:256 + 5] = 1
img_center_dft = center_fft(img_array)
img_center_slog = fft_slog(img_center_dft)

plt.imshow(img_center_slog, cmap='gray')
plt.show()
