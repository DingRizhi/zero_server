import logging
import math
from collections import namedtuple
import cv2
import numpy as np
from skimage import morphology
import wlib.libcom_pb2 as libcom__pb2


def make_segimg(image_np, top_x, top_y, bottom_x, bottom_y, box_add_length_to_mask):
    crop_width = bottom_x - top_x + box_add_length_to_mask * 2
    crop_height = bottom_y - top_y + box_add_length_to_mask * 2

    if top_x - box_add_length_to_mask < 0:
        left_x = 0
    else:
        left_x = top_x - box_add_length_to_mask

    if left_x + crop_width > image_np.shape[1]:
        left_x = image_np.shape[1] - crop_width

    if top_y - box_add_length_to_mask < 0:
        left_y = 0
    else:
        left_y = top_y - box_add_length_to_mask

    if left_y + crop_height > image_np.shape[0]:
        left_y = image_np.shape[0] - crop_height

    seg_img = image_np[left_y: left_y + crop_height,
              left_x:  left_x + crop_width,
              :]

    return seg_img, left_x, left_y


def _generate_mask(image_np, bbox):
    top_x, top_y, bottom_x, bottom_y = bbox[0], bbox[1], bbox[2], bbox[3]

    try:
        width = bottom_x - top_x
        height = bottom_y - top_y

        seg_img, left_x, left_y = make_segimg(image_np, top_x, top_y, bottom_x, bottom_y, box_add_length_to_mask=16)

        result_img = seg_img[:, :, 0]
        out_img = result_img[top_y - left_y:top_y - left_y + height, top_x - left_x:top_x - left_x + width]
        mask_img = out_img.astype(np.bool).astype(np.uint8) * 0

        if mask_img.shape[0] < 5 or mask_img.shape[1] < 5:
            logging.debug(mask_img.shape)
            mask_img = []
            points = []
        else:
            if np.sum(mask_img[:, :] > 0) == 0:
                mask_img = mask_img + 255
                mask_img[0, :] = 0
                mask_img[-1, :] = 0
                mask_img[:, 0] = 0
                mask_img[:, -1] = 0

            contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            points = _trans_list2points(contours[0].reshape((-1, 2)), top_x, top_y)
    except Exception as e:
        print(e)
        logging.debug(f"mask error {e}")
        mask_img = []
        points = []

    return mask_img, points


def mask2feature(r1):
    feature = _empty_feat()
    try:
        image_np = r1["image_np"]
        bbox = [int(r1["xmin"]), int(r1["ymin"]), int(r1["xmin"]) + int(r1["bb_width"]), int(r1["ymin"]) + int(r1["bb_height"]), r1["score"]]
        mask_img, points = _generate_mask(image_np, bbox)
        mask = np.asarray(bytearray(mask_img), dtype="uint8")

        if np.all(mask == 0):
            mask = None
        else:
            mask = mask.reshape(mask_img.shape)
        feature = _extract_feat(image_np, bbox, mask, limit_scale=100)

    except Exception as e:
        print(e)
        mask_img, points = [], []

    r1["length"] = feature['length']
    r1["width"] = feature['width']
    r1["pixel_area"] = feature['pixel_area']
    r1["gradients"] = feature['gradients']
    r1["contrast"] = feature['contrast']
    r1["brightness"] = feature['brightness']
    r1["max20brightness"] = feature['max20brightness']
    r1["min20brightness"] = feature['min20brightness']
    r1["points"] = list(points)
    r1["mask"] = mask_img.tobytes()


def _trans_list2points(point_list, x_offset, y_offset):
    """
    transform a point list to proto Point list
    """
    return list(map(lambda p: libcom__pb2.Point(x=p[0] + x_offset, y=p[1] + y_offset), point_list))


def _empty_feat():
    feature_result = {'length': 0,
                      'width': 0,
                      'pixel_area': 0,
                      'brightness': 0,
                      'max20brightness': 0,
                      'min20brightness': 0,
                      'gradients': 0,
                      'contrast': 0
                      }
    return feature_result


def _extract_feat(image_np, bbox, mask, limit_scale=100):
    image = image_np[bbox[1]: bbox[1] + mask.shape[0], bbox[0]: bbox[0] + mask.shape[1]]
    scale = 1
    if mask.shape[0] >= limit_scale or mask.shape[1] >= limit_scale:
        try:
            scale = limit_scale / max(mask.shape[0], mask.shape[1])
            new_w, new_h = int(image.shape[1] * scale), int(image.shape[0] * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask = mask.astype(np.bool)
            length, width = _extract_length_width_minarea(mask)
        except:
            logging.error(bbox)
            logging.error(image_np.shape)
            logging.error(image.shape)
            logging.error(scale)

    else:
        mask = mask.astype(np.bool)
        length, width = _extract_length_width_bydt(mask)

    length /= scale
    width /= scale

    pixel_area = _extract_pixel_area(mask)
    pixel_area = pixel_area / scale / scale
    brightness, max20brightness, min20brightness = _extract_brightness(mask, image)
    _gradients = _extract_gradients(mask, image)

    try:
        gradients = float(list(_gradients)[0])
    except:
        gradients = float(_gradients)[0]

    contrast = _extract_contrast(mask, image)
    feature_result = {'length': length,
                      'width': width,
                      'pixel_area': pixel_area,
                      'brightness': brightness,
                      'max20brightness': max20brightness,
                      'min20brightness': min20brightness,
                      'gradients': gradients,
                      'contrast': contrast
                      }
    return feature_result


def _extract_length_width_bydt(mask):
    """extract_length and width by distance transform"""
    height, width = mask.shape[0], mask.shape[1]
    # 使用骨架算法
    skeleton = morphology.skeletonize(mask)
    length = sum(skeleton.flatten())
    if length < min(height, width):
        length = min(height, width)  # 圆形缺陷的skeleton会被提取为一个点

    # distance transform
    dist_img = cv2.distanceTransform(mask.astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_3)
    width = np.median(dist_img[skeleton]) * 2

    return length, width


def _extract_length_width_minarea(mask):
    """extract_length and width using minarea algorithm"""
    """extract_length and width"""
    # opencv的旧版，返回三个参数，要想返回三个参数: pip install opencv-python==3.4.3.18,或者采用新版本
    # image, cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = mask.astype(np.uint8)
    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for cnt in cnts:
        areas.append(cv2.contourArea(cnt))
    index = np.argmax(areas)
    cnt = cnts[index]
    rect = cv2.minAreaRect(cnt)  # 最小外接矩形
    return rect[1][0], rect[1][1]


def _extract_length(mask, by_skeleton=True):
    """
    extract_length
    :param by_skeleton:
    :param mask: mask of target
    :return: length of the image _crop
    """
    height, width = mask.shape[0], mask.shape[1]
    if by_skeleton:
        # 使用骨架算法
        skeleton = morphology.skeletonize(mask)
        length = sum(skeleton.flatten())
        if length < min(height, width):
            length = min(height, width)  # 圆形缺陷的skeleton会被提取为一个点
        return length

    else:
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        xs = contours[0][0][:, 0]
        ys = contours[0][0][:, 1]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        # 计算目标区域中心点
        center_point = ((xmin + xmax) // 2, (ymax + ymin) // 2)
        # 找离中心点最远的点
        max_distance = 0
        for row in range(height):
            for col in range(width):
                if mask[row, col] == 0:
                    continue
                else:
                    distance = math.sqrt(((row - center_point[1]) ** 2) + ((col - center_point[0]) ** 2))
                    if distance > max_distance:
                        max_distance = distance
                        p1 = (row, col)
        # 找离p1最远的点
        length = 0
        for row in range(height):
            for col in range(width):
                if mask[row, col] == 0:
                    continue
                else:
                    distance_top1 = math.sqrt(((row - p1[1]) ** 2) + ((col - p1[0]) ** 2))
                    if distance_top1 > length:
                        length = distance_top1
                        p2 = (row, col)
    return length


def _extract_pixel_area(mask):
    """
    extract_pixel_area
    :param mask:
    :return:
    """
    return sum(sum(mask))


def _extract_brightness(mask, image):
    """
    extract_brightness
    :param mask:
    :param image:
    :return:
    """
    try:
        segm_pixels = image[mask == 1].flatten().tolist()
    except Exception as e:
        logging.debug('Mask shape: {}, image_shape: {}'.format(mask.shape, image.shape))
        return 0, 0, 0
    if len(segm_pixels) == 0:
        return 0, 0, 0

    top_k = max(1, int(len(segm_pixels) * 0.2))
    top_k_idx = sorted(segm_pixels, reverse=True)[0:top_k]
    low_k_idx = sorted(segm_pixels)[0:top_k]
    return sum(segm_pixels) / len(segm_pixels), sum(top_k_idx) / len(top_k_idx), sum(low_k_idx) / len(low_k_idx)


def _extract_gradients(mask, image):
    """
    extract_gradients
    :param mask:
    :param image:
    :return:
    """
    gray_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)  # x方向一阶导数
    gray_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)  # y方向一阶导数
    gradx = cv2.convertScaleAbs(gray_x)  # 转回原来的uint8形式
    grady = cv2.convertScaleAbs(gray_y)
    grad = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)  # 图像融合
    # 提取mask边缘点的梯度值
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 提取边缘点
    edge_points = []
    for contour in contours:
        for i in range(contour.shape[0]):
            edge_point = contour[i, 0, :]
            edge_points.append(edge_point)

    # 计算边缘点梯度均值
    grad_sum = 0
    for ep in edge_points:
        x, y = ep[0], ep[1]
        grad_sum += grad[y, x]
    return grad_sum if len(edge_points) == 0 else grad_sum / len(edge_points)


def _extract_contrast(mask, image, up_scale=100):
    """
    extract_contrast
    :param mask:
    :param image:
    :param up_scale:
    :return:
    """
    image_norm = image / 255
    fgs = image[mask != 0].flatten()
    bgs = image[mask == 0].flatten()
    if len(fgs) == 0:
        fg_mean = 0
    else:
        fg_mean = sum(fgs) / len(fgs)
    if len(bgs) == 0:
        bg_mean = 0
    else:
        bg_mean = sum(bgs) / len(bgs)

    contrast = abs(fg_mean - bg_mean)

    return contrast * up_scale
