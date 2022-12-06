import os
import argparse
import os
from tqdm import tqdm
from tools_utils.utils import *
import csv
import time
#从指定数据集中来切出缺陷

DEFECT_PROJECTION = {
    1: 'yashang',
    2: 'cashang',
    3: 'pengshang',
    4: 'huashang',
    5: 'madian',
    6: 'yise',
    7: 'yise',
    8: 'ok',
    9: 'zangwu',
    10: 'aotudian'}


class infer_record:
    # xy是左上角坐标
    def __init__(self, csv_row):
        self.fileName = str(csv_row[0])+".jpg"
        self.jsonName = str(csv_row[0])+".json"
        self.infer_defect = DEFECT_PROJECTION[int(csv_row[29])]
        self.score = float(csv_row[9])
        self.x = int(csv_row[4])
        self.y = int(csv_row[5])
        self.w = int(csv_row[6])
        self.h = int(csv_row[7])







def result_to_json(result_file_path: str, save_folder_path: str):
    '''
    :param json_folder_path: xml文件夹路径
    :param img_folder_path: 图像文件夹绝对路径
    :param selected: 要选取的缺陷名['pengshang'],如果有标签则转成pengshang_loushi,pengshang_guojian
    :return: 切出的img_size小图放在图片路径下
    '''


    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    all_result={}
    with open(result_file_path, 'r', newline='') as csvfile:
        # delimiter 指定分隔符，默认为逗号，这里指定为空格
        # quotechar 表示引用符
        # writerow 单行写入，列表格式传入数据
        spamreader = csv.reader(csvfile, delimiter=',', quotechar=' ')
        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            print(row)
            if int(row[25])==0: continue

            r = infer_record(row)
            if r.fileName not in all_result.keys():
                all_result[r.fileName]=[r]
            else:
                all_result[r.fileName].append(r)
            
           
    print(all_result)

    for key,items in all_result.items():

        json_dict = {"version": "4.2.10", "flags": {}, "shapes": [], "imagePath": items[0].fileName,
                     "imageData": None,
                     "imageHeight": None,
                     "imageWidth": None}
        for item in items:
            shape = {"label": item.infer_defect, "points": [[item.x, item.y],
                                                                [item.x + item.w,
                                                                 item.y + item.h]],
                    "group_id": {}, "shape_type": "rectangle","score":item.score,
                    "flags": {}}
            json_dict['shapes'].append(shape)
        json_fp = open(os.sep.join([save_folder_path, items[0].jsonName]), 'w')
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
        json_fp.close()



    


if __name__ == '__main__':
    # 填入xml folder path


    parser = argparse.ArgumentParser(prog='result_json.py')
    parser.add_argument('--result_file_path', default='../../inference/0101_Result.csv', help='path to resolve')
    parser.add_argument('--save_folder_path', default='../../inference/jsons/', help='path to test')

    opt = parser.parse_args()
    print(opt)


    result_to_json(opt.result_file_path, 
            opt.save_folder_path,
        )

