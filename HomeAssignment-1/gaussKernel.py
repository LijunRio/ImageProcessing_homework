import cv2
import numpy as np
from PIL import Image

# Read image
img = cv2.imread('einstein.tif')
img_array = np.array(img)
print("img_array: ", img_array.shape)
H, W, C = img.shape  # (679, 800, 3)
print("img_shape: ", img.shape)

# Gaussian Filter
K_size = 3
sigma = 1.3

# Zero padding
pad = K_size // 2  # 1 向下取整
# out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)  # (681, 802, 3)
# print("out_shape: ", out.shape)
# # 1:680 1:
# out[pad:pad + H, pad:pad + W] = img.copy().astype(np.float)


# replicated padding
out2 = np.zeros((H+pad*2, W+pad*2, C), dtype=np.float)
out2[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)
# 左列补齐
out2[0:pad, pad:-pad] = img_array[0:pad, :]
# 右列补齐
out2[-pad:, pad:-pad] = img_array[-pad:, :]
# 上下行补齐
out2[pad:-pad, 0:pad] = img_array[:, 0:pad]
out2[pad:-pad, -pad:] = img_array[:, -pad:]
# 四角元素补齐
out2[0,0] = img_array[pad, pad]
out2[-pad, pad]= img_array[-pad, pad]
out2[0, -pad] = img_array[0, -pad]
out2[-pad, -pad] = img_array[-pad, -pad]

out = out2


## Kernel
K = np.zeros((K_size, K_size), dtype=np.float)
for x in range(-pad, -pad + K_size):

    for y in range(-pad, -pad + K_size):
        K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
K /= (sigma * np.sqrt(2 * np.pi))
K /= K.sum()

tmp = out.copy()

for y in range(H):
    for x in range(W):
        for c in range(C):
            out[pad + y, pad + x, c] = np.sum(K * tmp[y:y + K_size, x:x + K_size, c])

out = out[pad:pad + H, pad:pad + W].astype(np.uint8)
print(out.shape)

cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
