from gaussBlur import gaussKernel
from gaussBlur import twodConv
from RgbToGray import rgb1gray
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def changeSigmaTest(sigma, m, padding, _img1, _img2, _img3, _img4):
    w1 = gaussKernel(sigma, m)
    result1 = twodConv(_img1, w1, padding)
    result2 = twodConv(_img2, w1, padding)
    result3 = twodConv(_img3, w1, padding)
    result4 = twodConv(_img4, w1, padding)

    plt.subplot(2, 2, 1)
    plt.title('einstein sigma=' + str(sigma))
    plt.imshow(result1, 'gray')

    plt.subplot(2, 2, 2)
    plt.title('cameraman sigma=' + str(sigma))
    plt.imshow(result2, 'gray')

    plt.subplot(2, 2, 3)
    plt.title('lena sigma=' + str(sigma))
    plt.imshow(result3, 'gray')

    plt.subplot(2, 2, 4)

    plt.title('mandril sigma=' + str(sigma))
    plt.imshow(result4, 'gray')
    plt.show()


def CompareWithOfficial(_img1, _img2, _img3, _img4, sigma, padding):
    kernel_size = round(3 * sigma * 2 + 1)
    blur1 = cv2.GaussianBlur(_img1, (kernel_size, kernel_size), sigma)
    blur2 = cv2.GaussianBlur(_img2, (kernel_size, kernel_size), sigma)
    blur3 = cv2.GaussianBlur(_img3, (kernel_size, kernel_size), sigma)
    blur4 = cv2.GaussianBlur(_img4, (kernel_size, kernel_size), sigma)

    w1 = gaussKernel(sigma, kernel_size)
    result1 = twodConv(_img1, w1, padding)
    result2 = twodConv(_img2, w1, padding)
    result3 = twodConv(_img3, w1, padding)
    result4 = twodConv(_img4, w1, padding)

    r1 = np.array(result1).astype(np.int32)  # 需要先进行类型转化，不然相减会出现0-255边界问题
    b1 = np.array(blur1).astype(np.int32)
    diff1 = abs(r1 - b1).astype(np.uint8)

    r2 = np.array(result2).astype(np.int32)
    b2 = np.array(blur2).astype(np.int32)
    diff2 = abs(r2 - b2).astype(np.uint8)

    r3 = np.array(result3).astype(np.int32)
    b3 = np.array(blur3).astype(np.int32)
    diff3 = abs(r3 - b3).astype(np.uint8)

    r4 = np.array(result4).astype(np.int32)
    b4 = np.array(blur4).astype(np.int32)
    diff4 = abs(r4 - b4).astype(np.uint8)

    plt.subplot(2, 2, 1)
    plt.title('einstein sigma=' + str(sigma))
    plt.imshow(diff1, 'gray')

    plt.subplot(2, 2, 2)
    plt.title('cameraman sigma=' + str(sigma))
    plt.imshow(diff2, 'gray')

    plt.subplot(2, 2, 3)
    plt.title('lena sigma=' + str(sigma))
    plt.imshow(diff3, 'gray')

    plt.subplot(2, 2, 4)
    plt.title('mandril sigma=' + str(sigma))
    plt.imshow(diff4, 'gray')

    plt.show()


def CompareTwoPadding(_img1, _img2, sigma):
    kernel_size = round(3 * sigma * 2 + 1)
    w1 = gaussKernel(sigma, kernel_size)
    result1 = twodConv(_img1, w1, 'replicate')
    result2 = twodConv(_img1, w1, 'zero')
    result3 = twodConv(_img2, w1, 'replicate')
    result4 = twodConv(_img2, w1, 'zero')

    plt.subplot(2, 2, 1)
    plt.title('lena sigma=' + str(sigma) + ' replicate')
    plt.imshow(result1, 'gray')

    plt.subplot(2, 2, 2)
    plt.title('lena sigma=' + str(sigma) + ' zero')
    plt.imshow(result2, 'gray')

    plt.subplot(2, 2, 3)
    plt.title('mandril sigma=' + str(sigma) + ' replicate')
    plt.imshow(result3, 'gray')

    plt.subplot(2, 2, 4)
    plt.title('mandril sigma=' + str(sigma) + ' zero')
    plt.imshow(result4, 'gray')

    plt.show()


# Read image
img1 = cv2.imread('einstein.tif', 0)  # f is grayscale
img2 = cv2.imread('cameraman.tif', 0)

# lena and mandril
img3 = cv2.imread('lena512color.tiff')
img3_gray = rgb1gray(img3, 'NTSC')
img4 = cv2.imread('mandril_color.tif')
img4_gray = rgb1gray(img4, 'NTSC')

# 改变sigma值进行对比
changeSigmaTest(1, 7, 'replicate', img1, img2, img3_gray, img4_gray)
CompareWithOfficial(img1, img2, img3_gray, img4_gray, 1, 'replicate')
CompareTwoPadding(img3_gray, img4_gray, 5)
