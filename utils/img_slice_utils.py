import random
from .utils import *
from loguru import logger



def img_crop(img, cx,cy,img_size):
    crop = center_crop_strategy(img, cx,cy, img_size)
    crop_size, fill_size = crop[0], crop[1]
    img_crop = img[crop_size[0]: crop_size[1], crop_size[2]: crop_size[3]]
    try:
        img_crop = img[crop_size[0]: crop_size[1], crop_size[2]: crop_size[3]]
    except Exception as e:
        logger.error(e)
        logger.error('fails in cropping , due to %s.\033[0m' % (e.with_traceback()))
        img_crop=None
    return img_crop

    
def get_crop_num(img_size, crop_size, overlap):
    '''
    :param img_size: img长或者宽
    :param crop_size: crop的边长
    :param overlap: 相邻框的交并比
    :return: 根据overlap和crop size计算滑框截取个数
    '''
    return math.ceil((img_size-crop_size)/((1-overlap)*crop_size)) + 1


def _random_crop(cx, cy, w, h, size, shift_x_left=0.75, shift_x_right=0.25, shift_y_up=0.75, shift_y_bottom=0.25):
    '''
    :param cx: 目标中心点x
    :param cy: 目标中心点y
    :param w: 图片width
    :param h: 图片height
    :param size: 截图的size
    :param shift_x_left: 截框左边框距离cx的最左随机范围（距离像素/size）
    :param shift_x_right: 截框左边框距离cx的最右随机范围（距离像素/size）
    :param shift_y_up: 截框上边框距离cy的最上随机范围（距离像素/size）
    :param shift_y_bottom: 截框上边框距离cy的最下随机范围（距离像素/size）
    :return: 返回随机截图框
    '''
    # 截框左边框、上边框距离目标中心点的offset
    ofx, ofy = random.randint(int(size*shift_x_right), int(size*shift_x_left)), random.randint(int(size*shift_y_bottom), int(size*shift_y_up))
    
    cx, cy = int(cx), int(cy)
    fill_size = [0, 0, 0, 0]
    if size > h:
        up, bottom = 0, h
        fill_size[0], fill_size[1] = (size-h)//2, size-h-(size-h)//2
    elif cy-ofy < 0:
        up, bottom = 0, size
    elif cy-ofy+size*0.5 > h:
        up, bottom = h-size, h
    elif cy-ofy-int(0.5*size) <0:
        up, bottom = 0, size
    else:
        up, bottom = cy-ofy-int(0.5*size), cy-ofy+int(size*0.5)
    if size > w:
        left, right = 0, w
        fill_size[2], fill_size[3] = (size-w)//2, size-w-(size-w)//2
    elif cx-ofx-0.5*size< 0:
        left, right = 0, size
    elif cx-ofx+0.5*size > w:
        left, right = w-size, w
    else:
        left, right = cx-ofx-int(0.5*size), cx-ofx+int(0.5*size)
    return [up, bottom, left, right], fill_size



# 检测特定标签的截图策略
def center_crop_strategy(img, cx,cy, size):
    # 是否目标居中
    centerness = True
    # 需要查看的labels
    crop_strategies = []
    w = img.shape[1]
    h = img.shape[0]
    return _random_crop(cx, cy, w, h, size, 0.0, 0.0, 0.0, 0.0)




def _combine_boxes(box1, box2):
    '''
    :param box1:
    :param box2:
    :return: 返回两个box的合并box
    '''
    xmin = min(box1.x, box2.x)
    ymin = min(box1.y, box2.y)
    xmax = max(box1.x+box1.w, box2.x+box2.w)
    ymax = max(box1.y+box1.h, box2.y+box2.h)
    return Box(xmin, ymin, xmax-xmin, ymax-ymin)

