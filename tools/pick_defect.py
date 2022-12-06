import os
import argparse
import os
from tqdm import tqdm
from tools_utils.utils import *
from tools_utils.img_slice_utils import img_crop
import logging
import time
#从指定数据集中来切出缺陷
def pick_defect(json_folder_path: str, img_folder_path: str,save_folder_path: str,selected:list, img_size:int):
    '''
    :param json_folder_path: xml文件夹路径
    :param img_folder_path: 图像文件夹绝对路径
    :param selected: 要选取的缺陷名['pengshang'],如果有标签则转成pengshang_loushi,pengshang_guojian
    :return: 切出的img_size小图放在图片路径下
    '''


    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    
    img_files = os.listdir(img_folder_path)
    img_list=[]
    # 遍历img
    for img_file in img_files:
        #print(img_file)
        img_file_path = os.path.join(img_folder_path, img_file)
        # 过滤文件夹和非图片文件
        if not os.path.isfile(img_file_path) or img_file[img_file.rindex('.')+1:] not in IMG_TYPES: continue
        # 对应的xml文件
        json_file_path = os.path.join(json_folder_path, img_file[:img_file.rindex('.')]+'.json')
        # 对应的json文件
        # json_out_path = os.path.join(img_folder_path, img_file[:img_file.rindex('.')]+'.json')

        if not os.path.isfile(json_file_path):
            continue
        image=cv2.imread(img_file_path)
        PassFlag=False
        try:
            if img_file[0:15] in img_list:
                #print("pass")
                PassFlag=True
        except:
            print("some error")
            PassFlag=False
            pass
        img_list.append(img_file[0:15])
        if PassFlag: continue
  

        #print(img_file_path)
        instance = json_to_instance(json_file_path)
        items = instance['shapes']
        for item in items:
            # print(item['shape_type'])
            # 如果有其他标签类别，通过elif添加
            
            if item['label'] not in selected:continue
            if item['shape_type'] != "rectangle": continue
            label = item['label']
            point1=item['points'][0]
            point2=item['points'][1]

            #print(label)
            img=img_crop(image,int((point1[0] +point2[0])*0.5),int((point1[1] +point2[1])*0.5),img_size)
                    #保存到results
            t=time.time()*1000
            class_name=label
            try:
                if len(item['tag'])>1:
                    class_name=label+"_"+item['tag']
            except:
                pass
            logging.debug("image {} class {} is in saved".format(img_file,class_name))

            try:
                jpg_name=img_file.split('.')[0][0:13]+str(round(t))+".jpg"
            except:
                jpg_name=img_file.split('.')[0]+str(round(t))+".jpg"
    
            if not os.path.exists(os.sep.join([save_folder_path,class_name])):
                os.mkdir(os.sep.join([save_folder_path,class_name]))  
            cv2.imwrite(os.sep.join([save_folder_path,class_name,jpg_name]),img)
            


if __name__ == '__main__':
    # 填入xml folder path


    parser = argparse.ArgumentParser(prog='pick_defect.py')
    parser.add_argument('--json_folder_path', default='/data/meboffline/lupinus_project/RYZK/datasets/hebing_2022-04-04_0122', help='path to test')
    parser.add_argument('--img_folder_path', default='/data/meboffline/lupinus_project/RYZK/datasets/hebing_2022-04-04_0122', help='path to test')
    parser.add_argument('--save_folder_path', default='../../inference/new/', help='path to test')
    parser.add_argument('--selected', default='huashang,pengshang', help='path to test')
    parser.add_argument('--img_size', default=224, help='path to test')
    opt = parser.parse_args()
    print(opt)
    if ',' in opt.selected:
        selected=opt.selected.split(',')
    else:
        selected=[opt.selected]

    pick_defect(opt.json_folder_path,
            opt.img_folder_path,
            opt.save_folder_path,
            selected,
            opt.img_size,
        )

