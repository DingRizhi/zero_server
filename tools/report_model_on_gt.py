from logging import exception
from tools_utils.utils import *
import glob
import os
import shutil
import csv
import argparse
from tools_utils.img_slice_utils import img_crop
import time

'''
Author: Yan Zhang
Date:2022/04/07
功能：   将；
        主要是用于生成过检和漏检标签,存为新的json文件
Parameters:
        model_json_path: 模型预测json文件路径   must
        gt_json_path:    实物核对标注文件路径    must
        gt_img_path:     实物核对图片路径       must
        output_path:     结果文件的存放目录路径  must

Return:
        None
'''

CSV_LINES = [['task_id', 'product_id', 'picture_id', 'gt_defect',
              'type', 'model_result', 'infer_defect', 'score', 'x', 'y', 'w', 'h']]
GROUP_ID_DIC_LUPINUS = {
    0: None,
    1: 'guojian',
    2: 'loushi',
    3: 'loushi',
    4: 'loushi'}
GROUP_ID_DIC_CSV = {
    0: 'right',
    1: 'guojian',
    2: 'loushi',
    3: 'loushi_weak',
    4: 'loushi_nosee'}


class infer_record:
    # xy是左上角坐标
    def __init__(self, csv_row):
        self.task_id = int(csv_row[0])
        self.product_id = int(csv_row[1])
        self.picture_id = int(csv_row[2])
        self.gt_defect = str(csv_row[3])
        self.type = str(csv_row[4])
        self.model_type = str(csv_row[5])
        self.infer_defect = str(csv_row[6])
        self.score = float(csv_row[7])
        self.x = int(csv_row[8])
        self.y = int(csv_row[9])
        self.w = int(csv_row[10])
        self.h = int(csv_row[11])

    def get_product(self):
        product_name = "{:0>4d}-{:0>4d}".format(self.task_id, self.product_id)
        return product_name

    def get_pic_name(self):
        pic_name = "{:0>4d}-{:0>4d}-{:0>2d}.jpg".format(
            self.task_id, self.product_id, self.picture_id)
        return pic_name


def np_to_csv(var, save_path):
    f = open(os.sep.join([save_path, 'result.csv']), 'w', newline='')
    writer = csv.writer(f)
    for i in var:
        writer.writerow(i)
    f.close()


def label_comparison(gt_img_path, gt_json_path, model_json_path, output_path, save_img,
                     iou_thres=0.01, guojian_thres=0):  # normal,guojian,loushi
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.sep.join([output_path,"imgs"])):
        os.mkdir(os.sep.join([output_path,"imgs"]))
    if not os.path.exists(os.sep.join([output_path,"jsons"])):
        os.mkdir(os.sep.join([output_path,"jsons"]))
    assert os.path.exists(gt_img_path) and os.path.exists(gt_img_path) and os.path.exists(gt_img_path),"Path not exist"
      

    imgs = glob.glob(os.sep.join([gt_img_path, "*.jpg"]))
    ground_truth_jsons = glob.glob(os.sep.join([gt_json_path, "*.json"]))
    prediction_jsons = glob.glob(os.sep.join([model_json_path, "*.json"]))

    for img in imgs:
        img_name = os.path.basename(img)
        img_name_without_suffix = img_name.split('.')[0]

        copy_to = os.sep.join([output_path, "imgs",img_name])

        gtj = None
        for i in range(len(ground_truth_jsons)):
            gtj_tmp = os.path.basename(ground_truth_jsons[i])
            gtj_name = gtj_tmp.split('.')[0]
            if gtj_name == img_name_without_suffix:
                gtj = ground_truth_jsons[i]
                break

        pj = None
        for i in range(len(prediction_jsons)):
            pj_tmp = os.path.basename(prediction_jsons[i])
            pj_name = pj_tmp.split('.')[0]
            if pj_name == img_name_without_suffix:
                pj = prediction_jsons[i]
                break

        guojian_instance = []
        loushi_instance = []
        normal_instance = []
        cuowu_instance = []
        used = []

        task_id, product_id, picture_id = img_name_without_suffix.split('-')

        # if gtj is None or pj is None: continue
        if gtj is None and pj is None:
            csv_line = [
                    task_id,
                    product_id,
                    picture_id,
                    "",
                    "NULL",
                    "right",
                    '',
                    '0.0',
                    0,
                    0,
                    0,
                    0]
            CSV_LINES.append(csv_line)
            continue
        elif gtj is  None and pj is not None:
            print("here")
            pj_instance = json_to_instance(pj)
            pj_shapes = pj_instance['shapes']
            filtered_pj_shapes = []
            for pj_shape in pj_shapes:
                if pj_shape['score'] >= guojian_thres:
                    filtered_pj_shapes.append(pj_shape)
            pj_shapes = filtered_pj_shapes 
            for i, pj_shape in enumerate(pj_shapes):
                x, y, w, h = points_to_xywh(pj_shape)
                csv_line = [
                    task_id,
                    product_id,
                    picture_id,
                    "",
                    'guojian',
                    'guojian',
                    pj_shape['label'],
                    pj_shape['score'],
                    x,
                    y,
                    w,
                    h]
                CSV_LINES.append(csv_line)    
            gtj_instance=pj_instance
            gtj_instance['shapes']=pj_shapes
            for gtj_shape in gtj_instance['shapes']:
                gtj_shape['tag'] = "guojian"
           

        elif gtj is not None and pj is None:
            gtj_instance = json_to_instance(gtj)
            gtj_shapes = gtj_instance['shapes']

            for gtj_shape in gtj_shapes:
                # gtj_shape['group_id'] = 2
                gtj_shape['tag'] = 'loushi'
                csv_tag = 'loushi'
                x, y, w, h = points_to_xywh(gtj_shape)
                csv_line = [
                    task_id,
                    product_id,
                    picture_id,
                    gtj_shape['label'],
                    csv_tag,
                    csv_tag,
                    '',
                    '0.0',
                    x,
                    y,
                    w,
                    h]
                CSV_LINES.append(csv_line)

        elif gtj is not None and pj is not None:
            gtj_instance = json_to_instance(gtj)
            pj_instance = json_to_instance(pj)
            gtj_shapes = gtj_instance['shapes']
            pj_shapes = pj_instance['shapes']

            filtered_pj_shapes = []
            for pj_shape in pj_shapes:
                if pj_shape['score'] >= guojian_thres:
                    filtered_pj_shapes.append(pj_shape)
            pj_shapes = filtered_pj_shapes

            for gtj_shape in gtj_shapes:
                x, y, w, h = points_to_xywh(gtj_shape)
                if 'group_id' not in gtj_shape.keys():
                    gtj_shape['group_id'] = 0
                if gtj_shape['group_id']==None or gtj_shape['group_id']=='' or gtj_shape['group_id']=={}:
                    gtj_shape['group_id'] = 0
                gt_box = Box(x, y, w, h, gtj_shape['label'])
                false_negative, false_label, false_conf, hard, hard_conf = True, True, 0, True, 0
                for i, pj_shape in enumerate(pj_shapes):
                    x1, y1, w1, h1 = points_to_xywh(pj_shape)
                    p_box = Box(
                        x1,
                        y1,
                        w1,
                        h1,
                        pj_shape['label'],
                        pj_shape['score'])
                    if gt_box.get_iou(p_box) > iou_thres:
                        used.append(i)
                        false_negative = False
                        # gtj_shape['score'] = pj_shape['score']
                        # print(gtj_shape)
                        if gt_box.category == p_box.category:
                            normal_instance.append(gtj_shape)
                        else:
                            normal_instance.append(gtj_shape)
                if false_negative:
                    # gtj_shape['label'] = gtj_shape['label'] + '_loushi'
                    gtj_shape['tag'] = 'loushi'
                    csv_tag = GROUP_ID_DIC_CSV[gtj_shape['group_id']]
                    csv_line = [
                        task_id,
                        product_id,
                        picture_id,
                        gtj_shape['label'],
                        csv_tag,
                        'loushi',
                        '',
                        '0.0',
                        x,
                        y,
                        w,
                        h]
                else:
                    # gtj_shape['label'] = gtj_shape['label'] + '_normal'
                    if gtj_shape['group_id'] == 1:
                        gtj_shape['tag'] = 'guojian'
                        csv_tag = GROUP_ID_DIC_CSV[gtj_shape['group_id']]
                        csv_line = [
                            task_id,
                            product_id,
                            picture_id,
                            gtj_shape['label'],
                            csv_tag,
                            'guojian',
                            gtj_shape['label'],
                            pj_shape['score'],
                            x,
                            y,
                            w,
                            h]
                    else:
                        gtj_shape['tag'] = None
                        csv_tag = GROUP_ID_DIC_CSV[gtj_shape['group_id']]
                        csv_line = [
                            task_id,
                            product_id,
                            picture_id,
                            gtj_shape['label'],
                            csv_tag,
                            'right',
                            gtj_shape['label'],
                            pj_shape['score'],
                            x,
                            y,
                            w,
                            h]
                CSV_LINES.append(csv_line)
            for index, pj_shape in enumerate(pj_shapes):
                if index not in used:
                    # pj_shape['label'] = pj_shape['label'] + '_guosha'
                    x, y, w, h = points_to_xywh(pj_shape)
                    pj_shape['tag'] = 'guojian'
                    pj_shape['group_id'] = 1
                    gtj_instance['shapes'].append(pj_shape)
                    csv_tag = 'guojian'
                    csv_line = [
                        task_id,
                        product_id,
                        picture_id,
                        '',
                        csv_tag,
                        csv_tag,
                        pj_shape['label'],
                        pj_shape['score'],
                        x,
                        y,
                        w,
                        h]
                    CSV_LINES.append(csv_line)
        try:
            json_file = os.path.basename(gtj)
        except:
            json_file = os.path.basename(pj)

        if save_img:
            shutil.copyfile(img, copy_to)
        instance_to_json(gtj_instance, os.sep.join([output_path,"jsons",json_file]))
    np_to_csv(CSV_LINES, output_path)


def output_img(csv_result_path, img_path,output_path, img_size):
    with open(csv_result_path, 'r', newline='') as csvfile:
        # delimiter 指定分隔符，默认为逗号，这里指定为空格
        # quotechar 表示引用符
        # writerow 单行写入，列表格式传入数据
        spamreader = csv.reader(csvfile, delimiter=',', quotechar=' ')
        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            print(row)
            r = infer_record(row)
            img_file_path = os.sep.join([img_path, r.get_pic_name()])
            if not os.path.isfile(img_file_path):
                continue

            if r.model_type == "right" and r.type == "guojian":
                type = "guojian"
            if r.model_type == "right" and r.type != "guojian" and r.type != "NULL":
                type = "right"
            elif r.model_type == "guojian":
                type = "guojian"
            elif r.model_type == "loushi" and r.type == "loushi_weak":
                type = "loushi_weak"
            elif r.model_type == "loushi" and r.type == "loushi_nosee":
                type = "loushi_nosee"
            elif r.model_type == "loushi":
                type = "loushi"
            else:
                type = None
            if not type:
                continue
            image = cv2.imread(img_file_path)

            img = img_crop(image, int(r.x + r.w * 0.5),
                           int(r.y + r.h * 0.5), img_size)
            # 这里切图加入了时间戳
            t = time.time() * 1000
            if r.infer_defect:
                class_name = r.infer_defect
            else:
                class_name = r.gt_defect
            jpg_name = r.get_pic_name().split('.')[0] + str(round(t)) + ".jpg"
            if not os.path.exists(output_path):
                os.mkdir(os.sep.join(output_path))
            if not os.path.exists(os.sep.join([output_path, class_name])):
                os.mkdir(os.sep.join([output_path, class_name]))
            if not os.path.exists(os.sep.join([output_path, class_name, type])):
                os.mkdir(os.sep.join([output_path, class_name, type]))
            cv2.imwrite(os.sep.join(
                [output_path, class_name, type, jpg_name]), img)


def report(csv_result_path,output_path):
    all_result={}
    with open(csv_result_path, 'r', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar=' ')
        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            print(row)
            r = infer_record(row)

            if r.model_type == "right" and r.type == "guojian":
                type = "guojian"
            elif r.model_type == "right" and r.type == "NULL":
                type = "liangpin"
            elif r.model_type == "right" and r.type != "guojian" and r.type != "NULL":
                type = "right"
            elif r.model_type == "guojian":
                type = "guojian"
            elif r.model_type == "loushi" and r.type == "loushi_weak":
                type = "loushi"
            elif r.model_type == "loushi" and r.type == "loushi_nosee":
                type = "loushi"
            elif r.model_type == "loushi":
                type = "loushi"
            else:
                type = None
            if not type:
                continue
            if r.get_product() in all_result.keys():
                all_result[r.get_product()][type]+=1
            else:
                all_result[r.get_product()]={'right':0,'loushi':0,'guojian':0,'liangpin':0}
    print(all_result)
    f = open(os.sep.join([output_path, 'product_result.csv']), 'w', newline='')
    csv_head=['Product','Final_Confirm','Detail_loushi','Detail_guojian','Detail_right','Detail_liangpin']
    writer = csv.writer(f)
    writer.writerow(csv_head)
    for key,item in all_result.items():
        type='right'
        if item['loushi']==0 and item['guojian']==0 and item['right']==0:
            type='right_liangpin'
        elif item['loushi']>0 and item['guojian']>0 and item['right']==0:
            type='right_cuojian'
        elif item['loushi']>0 and item['guojian']==0 and item['right']==0:
            type='loushi'
        elif item['guojian']>0 and item['right']==0 and item['loushi']==0 :
            type="guojian"
        elif item['right']>0:
            type='right'
        elif item['liangpin']>0 and item['loushi']==0 and item['guojian']==0 and item['right']==0:
            type='right_liangpin'
        
        writer.writerow([key,type,item['loushi'],item['guojian'],item['right'],item['liangpin']])
        print([key,type,item['loushi'],item['guojian'],item['right'],item['liangpin']])
    f.close()
            



if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='report_model_on_gt.py')
    parser.add_argument(
        '--gt_img_path',
        default='../../inference/v1/image',
        help='path to test')
    parser.add_argument(
        '--gt_json_path',
        default='../../inference/v1/gt',
        help='path to test')
    parser.add_argument(
        '--model_json_path',
        default='../../inference/v1/model',
        help='path to test')
    parser.add_argument(
        '--output_path',
        default='../../inference/v1/output',
        help='path to test')
    parser.add_argument(
        '--img_size',
        default=224,
        help='crop image size')
    parser.add_argument(
        '--crop_img',
        default=True,
        help='crop image')
    parser.add_argument(
        '--save_origin',
        default=True,
        help='save origin image')
        
    opt = parser.parse_args()
    print(opt)
    iou_thres = 0.01
    guojian_thres = 0.1

    label_comparison(opt.gt_img_path, opt.gt_json_path, opt.model_json_path, opt.output_path,opt.save_origin, iou_thres,
                     guojian_thres)

    csv_file = os.sep.join([opt.output_path, 'result.csv'])
    if opt.crop_img:
        output_img(csv_file, opt.gt_img_path,opt.output_path, opt.img_size)
    report(csv_file,opt.output_path)
