import cv2
import numpy as np

from binary import Binaryzation
from skeleton import mor_skeleton
from skeleton import distTrans_skeleton
from tailor import thinning
from tailor import tailor


def main(img):

    # 二值化: 基于迭代法获取二值化阈值
    print("Processing: 二值化...")
    threshold, k, img_bin = Binaryzation(np.array(img))

    print("阈值:", threshold)
    cv2.imwrite("./result/img_binary.png", img_bin)

    # 基于形态学的骨架提取./result/
    print("Processing: 基于形态学的骨架提取...")
    img_sk_morph = mor_skeleton(img_bin)
    cv2.imwrite("./result/morph_skeleton.png", img_sk_morph)

    # 基于距离变换的骨架提取
    print("Processing: 基于距离变换的骨架提取...")
    img_sk_dist = distTrans_skeleton(img_bin)
    cv2.imwrite("./result/distTrain_skeleton.png", img_sk_dist)

    # 图像细化
    print("Processing: 图像细化...")
    img_sk_thin = thinning(img_bin)
    cv2.imwrite("./result/thinning.png", img_sk_thin)

    # 裁剪:以细化所得骨架为例
    print("Processing: 裁剪:以细化所得为例...")
    img_result = tailor(img_sk_thin)
    cv2.imwrite("./result/tailor.png", img_result)


if __name__ == "__main__":
    img = cv2.imread("./img/fingerprint.jpg", 0)
    main(img)