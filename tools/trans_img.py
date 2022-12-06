import argparse
import glob
import cv2
import os
from tqdm import tqdm
from tools_utils.utils import *
def adaptivehistogram_enhance(img, clipLimit=2.0, tileGridSize=(8, 8)):
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

def trans(origin_path,target_path,img_size):

    img_items = glob.glob(os.sep.join([origin_path,"*.jpg"]))
    pimg_items=tqdm(img_items)
    img_num=len(img_items)
    for img_item in pimg_items:
        
        img = cv2.imread(img_item)
        jpg_name = os.path.basename(img_item)
        
        (p,fn)=os.path.split(img_item)
       
        flag, img = adaptivehistogram_enhance(
            img, clipLimit=100.0, tileGridSize=(8, 8))
        img=cv2.resize(img,(img_size,img_size))
     
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        to_jpg_path1 = os.sep.join([target_path, jpg_name])
        cv2.imwrite(to_jpg_path1, img)
        #print(to_jpg_path1)
        #开始处理标注文件
        json_file_path = os.sep.join([origin_path, jpg_name[:jpg_name.rindex('.')]+'.json'])
        json_out_path= os.sep.join([target_path, jpg_name[:jpg_name.rindex('.')]+'.json'])
        if os.path.isfile(json_file_path):
            instance = json_to_instance(json_file_path)
            w=instance['imageWidth']
            h=instance['imageHeight']
            items = instance['shapes']
            instance['imageData'] = None
            instance['imageHeight']=img_size
            instance['imageWidth']=img_size
            new_items = []
            for item in items:
                # print(item['shape_type'])
                # 如果有其他标签类别，通过elif添加
                #print(label)
                rec_points=[]
                xys = item['points']
                for p in xys:
                    rec_points.append([int(p[0]*img_size/w),int(p[1]*img_size/h)])
                item['points'] = rec_points
                new_items.append(item)
            instance['shapes']=new_items
            instance_to_json(instance, json_out_path)
            #print('Json created:', instance)
        pimg_items.set_description("Processing %s" % jpg_name)
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='trans_img.py')
    parser.add_argument('--origin', default=r'C:\Users\yuanw\inference\test', help='path to test')
    parser.add_argument('--target', default=r'C:\Users\yuanw\inference\new', help='path to test')
    parser.add_argument('--img_size', default=1024, help='resize to this size')

    opt = parser.parse_args()
    print(opt)
    trans(opt.origin,
            opt.target,
            opt.img_size
        )
