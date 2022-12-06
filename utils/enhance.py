# 图像预处理

# sharpen(img):图像锐化
# sp_noise(img,prob):椒盐噪声
# gasuss_noise(img, mean=0, var=0.001):高斯噪声
# gamma(img, c, v):伽马变换
# GaussianBlurring(img,k):高斯模糊
# equalizeHist(img):直方图均衡
# adaptivehistogram(img,clipLimit=2.0, tileGridSize=(8,8)):自适应直方图均衡
# Contrast_and_Brightness(img, alpha, beta):亮度变化


# -*- coding:UTF-8 -*-
import glob
import numpy as np
import cv2
import random
import os


# 图像锐化
# input:图像
# output:通道数flag（反映图像是否为空），图像
def sharpen_enhance(img):
    flag = 0
    if img is None:
        print('图像为空')
        return flag, img
    if len(img.shape) == 2:
        flag = 1
    if len(img.shape) == 3:
        flag = 3
    if img.sum() == 0:
        print('warning:全为零')

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    img = cv2.filter2D(img, -1, kernel=kernel)
    # print(img.shape)
    return flag, img


# 椒盐噪声
# input:图像，噪声比例prob
# output:通道数flag（反映图像是否为空），图像
def sp_noise_enhance(img, prob):
    flag = 0
    if img is None:
        print('图像为空')
        return flag, img
    if len(img.shape) == 2:
        flag = 1
    if len(img.shape) == 3:
        flag = 3
    if img.sum() == 0:
        print('warning:全为零')

    output = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    # print(output.shape)
    return flag, output


# 高斯噪声
# input:图像，均值mean，方差var
# output:通道数flag（反映图像是否为空），图像
def gasuss_noise_enhance(img, mean=0, var=0.001):
    flag = 0
    if img is None:
        print('图像为空')
        return flag, img
    if len(img.shape) == 2:
        flag = 1
    if len(img.shape) == 3:
        flag = 3
    if img.sum() == 0:
        print('warning:全为零')

    image = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    # print(out.shape)
    return flag, out


# 伽马变换
# input:图像，倍数c，指数v
# output:通道数flag（反映图像是否为空），图像
def gamma_enhance(img, c, v):
    flag = 0
    if img is None:
        print('图像为空')
        return flag, img
    if len(img.shape) == 2:
        flag = 1
    if len(img.shape) == 3:
        flag = 3
    if img.sum() == 0:
        print('warning:全为零')

    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
        if lut[i] >= 254:
            lut[i] = 254
    output_img = cv2.LUT(img, lut)  # 像素灰度值的映射
    output_img = np.uint8(output_img + 0.5)
    # print(output_img.shape)
    return flag, output_img


# 高斯模糊
# input:图像，核的大小k
# output:通道数flag（反映图像是否为空），图像
def GaussianBlurring_enhance(img, k):
    flag = 0
    if img is None:
        print('图像为空')
        return flag, img
    if len(img.shape) == 2:
        flag = 1
    if len(img.shape) == 3:
        flag = 3
    if img.sum() == 0:
        print('warning:全为零')

    img = cv2.GaussianBlur(img, (k, k), 0)
    print(img.shape)
    return flag, img


# 直方图均衡
# input:图像
# output:通道数flag（反映图像是否为空），图像
def equalizeHist_enhance(img):
    flag = 0
    if img is None:
        print('图像为空')
        return flag, img
    if img.sum() == 0:
        print('warning:全为零')
    if len(img.shape) == 2:
        flag = 1
        equ = cv2.equalizeHist(img)
    if len(img.shape) == 3:
        flag = 3
        b, g, r = cv2.split(img)
        img1 = cv2.equalizeHist(b)
        img2 = cv2.equalizeHist(g)
        img3 = cv2.equalizeHist(r)
        equ = cv2.merge([img1, img2, img3])

    return flag, equ


# 自适应直方图均衡
# input:图像，clipLimit，tileGridSize
# output:通道数flag（反映图像是否为空），图像
def adaptivehistogram_enhance(img, clipLimit=100.0, tileGridSize=(8, 8)):
    flag = 0
    if img is None:
        print('图像为空')
        return flag, img
    if img.sum() == 0:
        print('warning:全为零')

    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    if len(img.shape) == 2:
        flag = 1
        cl1 = clahe.apply(img)
    if len(img.shape) == 3:
        flag = 3
        b, g, r = cv2.split(img)
        img1 = clahe.apply(b)
        img2 = clahe.apply(g)
        img3 = clahe.apply(r)
        cl1 = cv2.merge([img1, img2, img3])

    return flag, cl1


# 亮度变化
# input:图像，，α调节对比度， β调节亮度
# output:通道数flag（反映图像是否为空），图像
def Contrast_and_Brightness_enhance(img, alpha, beta):
    flag = 0
    if img is None:
        print('图像为空')
        return flag, img
    if len(img.shape) == 2:
        flag = 1
    if len(img.shape) == 3:
        flag = 3
    if img.sum() == 0:
        print('warning:全为零')

    blank = np.zeros(img.shape, img.dtype)
    dst = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)
    print(dst.shape)
    return flag, dst


# 旋转angle角度，缺失背景白色（255, 255, 255）填充
def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(0, 0, 0))
    # borderValue 缺省，默认是黑色（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))


def contrast_brightness_image(src1, a, g):
    h, w, ch = src1.shape  # 获取shape的数值，height和width、通道

    # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)
    dst = cv2.addWeighted(src1, a, src2, 1-a, g)  # addWeighted函数说明如下
    # cv2.imshow("con-bri-demo", dst)
    return dst


if __name__ == '__main__':

    sourse_image_list = glob.glob("/home/adt/mnt/data/吕挤线/ok/*jpg")
    convert_path = "/home/adt/mnt/data/吕挤线/ok_convert"
    if not os.path.exists(convert_path):
        os.makedirs(convert_path)
    for img_p in sourse_image_list:
        img_name = os.path.basename(img_p)
        img_convert_p = os.path.join(convert_path, img_name)

        img = cv2.imread(img_p)
        flag, cl1 = adaptivehistogram_enhance(
            img, clipLimit=100.0, tileGridSize=(8, 8))

        cv2.imwrite(img_convert_p, cl1)
        print(img_convert_p)

    # flag, cl1 = equalizeHist_enhance(img)

    # cl1 = contrast_brightness_image(img, 2, 1)

    # jpg_path = glob.glob(
    #     "/home/kerry/mnt/mark/apple575/p05/now/train/20210522/damian/pr/liewen/*.jpg")
    # json_str = "/home/kerry/mnt/mark/apple575/p05/now/train/20210522/damian/pr/liewen/*.json"
    # enhance_path = '/home/kerry/mnt/mark/apple575/p05/now/train/20210522/damian/pr/liewen_enhance/'

    # if not os.path.exists(enhance_path):
    #     os.makedirs(enhance_path)

    # step = 1
    # light = 1.2

    # for jpg_p in jpg_path:
    #     print(jpg_p)
    #     jpg_name = os.path.basename(jpg_p)
    #     img = cv2.imread(jpg_p)
    #     angle = 1.0
    #     flag, img_new = Contrast_and_Brightness_enhance(img, light, light)

    #     to_jpg_path = os.path.join(enhance_path, str(step) + '/')
    #     if not os.path.exists(to_jpg_path):
    #         os.makedirs(to_jpg_path)
    #     cmd_cp = "cp " + json_str + " " + to_jpg_path
    #     os.system(cmd_cp)

    #     to_jpg_path1 = os.path.join(to_jpg_path, jpg_name)

    #     cv2.imwrite(to_jpg_path1, img_new)
