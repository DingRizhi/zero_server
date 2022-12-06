# -*- coding:UTF-8 -*-
import sys


sys.path.append('../')

import math
import os
import pprint
import time
import traceback
from concurrent.futures._base import wait
from concurrent.futures.thread import ThreadPoolExecutor
import cv2
from queue import Queue
import logging
import grpc
import numpy as np
import torch
import yaml
import rule_engine
from PIL import Image
from torch.nn import functional
from torchvision import transforms
from models.experimental import attempt_load
from utils.enhance import adaptivehistogram_enhance
from wlib.libcom_cls import basic_classifier_mobilenetv3,basic_classifier_resnet50,\
    basic_classifier_densenet121,basic_classifier_efficientnet_b0,basic_classifier_mobilenetv2,\
    basic_classifier_mobilenetv3_small, basic_classifier_resnet18_v2
from wlib.libcom_pb2 import InferenceReply, SingleInferenceReply
from wlib.libcom_pb2_grpc import InferenceServicer, add_InferenceServicer_to_server
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from wlib.libfeature import mask2feature
from utils.img_slice_utils import img_crop
import copy
BACKBONE_DICT = {'mobilenetv3s':basic_classifier_mobilenetv3_small,
                'mobilenetv3':basic_classifier_mobilenetv3,
                'resnet50':basic_classifier_resnet50,
                'mobilenetv2':basic_classifier_mobilenetv2,
                'densenet':basic_classifier_densenet121,
                'efficientnet':basic_classifier_efficientnet_b0,
                'resnet18v2':basic_classifier_resnet18_v2
                }


project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def enhance_resize(image_np, clipLimit, pre_size):
    if clipLimit > 0:
        clipLimit_in = float(clipLimit)
        _, image_np1 = adaptivehistogram_enhance(
            image_np, clipLimit=clipLimit_in, tileGridSize=(8, 8))
        image_np2 = cv2.resize(image_np1, (pre_size, pre_size))
    #print(image_np2.shape)
    return image_np2


def server_start(config):
    ip = config['global_config']['ip']
    port = config['global_config']['port']
    num_workers = config['global_config']['num_workers']
    address = f'{ip}:{port}'
    options = (
        ('grpc.max_send_message_length', 128 * 1024 * 1024), ('grpc.max_receive_message_length', 128 * 1024 * 1024),)
    server = grpc.server(ThreadPoolExecutor(max_workers=num_workers), options=options)
    add_InferenceServicer_to_server(Inference(config=config), server)
    server.add_insecure_port(address)
    server.start()
    logging.info('server starts successfully at {}'.format(address))
    print(f'server starts successfully at {address}')
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)


class Inference(InferenceServicer):

    def __init__(self, config):
        super(Inference, self).__init__()
        self.det_models = [init_model(info) for info in config['det_models']]
        self.cls_models = [init_model(info) for info in config['cls_models']]
        self.post_cls_models=[init_post_model(info) for info in config['post_cls_models']]
        print(f'Models initialized Successfully!')

    def Inference(self, request, context):
        # 开始计时
        logging.info('Receive inference request with photo id :{}, '
                     'produce id :{}, channel id :{}.'.format(request.photo_id,
                                                              request.product_id,
                                                              request.channel_id))
        t0 = time.time()
        reply = InferenceReply(num_detections=0, photo_id=request.photo_id, product_id=request.product_id,
                               channel_id=request.channel_id)
        # 读入图片
        img_bytes = bytearray(request.encoded_image)
        image = np.asarray(img_bytes, dtype='uint8').reshape((request.height, request.width, -1))
        # 图片shape必须为(width, height, 3)
        if image.shape[2] == 1:
            image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        try:
            enhance_id=[]
            for dm in self.det_models:
                if dm['enhance']==True:
                    enhance_id.extend(dm['photo_ids'])
            for cm in self.cls_models:
                if cm['enhance']==True:
                    enhance_id.extend(cm['photo_ids'])
            logging.info('Inference start.')
            if request.photo_id in enhance_id:
                image_enhance_temp = enhance_resize(image, 1024, 1024)
                image_enhance=copy.deepcopy(image_enhance_temp)
            else:
                image_enhance=None
            image_name="{:0>4d}-{:0>4d}-{:0>2d}.jpg".format(request.channel_id,request.product_id,request.photo_id)
            singles = inference(self.det_models, self.cls_models,self.post_cls_models, image, image_enhance, request.photo_id,image_name)
        except Exception as e:
            logging.fatal("Image {} Fail to inference a image as Exception: {}".format(image_name,e))
            traceback.print_exc()
            return reply
        # 结束计时
        logging.info(f'Image {image_name} inference process finished in {time.time() - t0} seconds.')
        print(f'Image {image_name} Inference Process Finished in {time.time() - t0} seconds.')
        reply.num_detections = len(singles)
        reply.singleInferenceReply.extend(singles)
        print(singles)
        return reply


def init_model(info):
    '''
    :param info: 信息字典
    :return: 模型字典
    '''
    model_dict = {'class_dict': info['class_dict'], 'photo_ids': info['photo_ids'], 'conf_thres': info['conf_thres']}
    # 分类模型的信息字典中有backbone字段，如‘resnet18’，‘mobilenetv2’
    cls_model = 'backbone' in info.keys()
    model_dict['model'] = Queue(maxsize=0)

    if cls_model:
        model_dict['enhance'] =info['enhance']
        device_id = info['device_id']
        device = torch.device(f'cuda:{device_id}')
        model_dict['resize']=info['resize']
        model_dict['device'] = device
        model_dict['filter'] =info['filter']
        for _ in range(info['model_copies']):
            if info['backbone'] in BACKBONE_DICT.keys():
                model = BACKBONE_DICT[info['backbone']](info['num_classes'])
                model.load_state_dict(torch.load(info['weight_path']), False)
                model_dict['transform'] = transforms.Compose([
                    transforms.Resize(info['img_size']),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
                    ])
                device_id = info['device_id']
                device = torch.device(f'cuda:{device_id}')
                model.to(device)
                model.eval()
                #model.half()
            else:
                model_dict['transform'] = transforms.Compose([
                    transforms.Resize(info['img_size']),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
                    ])
                model = torch.load(info['weight_path'])
                device_id = info['device_id']
                device = torch.device(f'cuda:{device_id}')
                model.to(device)
                model.eval()
            print("loading cls model {}".format(info['weight_path']))
            model_dict['model'].put(model)
        # transform = transforms.Compose([transforms.Resize(info['img_size']),
        #                                 transforms.ToTensor(),
        #                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            model_dict['class_ok_name'] = info['class_ok_name']
    else:
        models=info['device_id']
        model_dict['enhance'] =info['enhance']

        try:
            model_dict['analyze_defects'] = info['analyze_defects']
        except:
            model_dict['analyze_defects'] = []

        model_dict['resize'] =info['resize']
        for m in models.keys():
           for i in range(int(models[m])):
                device_id = m
                device1= torch.device(f'cuda:{device_id}')
                model1 = attempt_load(info['weight_path'])
                model1.to(device1)
                model1.half()
                model1.eval()
                model_dict['model'].put([device1,model1])
        model_dict['iou_thres'] = info['iou_thres']
        model_dict['filter'] =info['filter']
    if info['filter']:
        with open(model_dict['filter'], 'r') as file_h:
            model_dict['filter']= yaml.load(file_h, Loader=yaml.FullLoader)
    return model_dict





def init_post_model(info):
    '''
    :param info: 信息字典
    :return: 模型字典
    '''
    model_dict = {'defect': info['defect'],'img_size':info['img_size'],'class_dict':info['class_dict']}
    # 分类模型的信息字典中有backbone字段，如‘resnet18’，‘mobilenetv2’
    model_dict['model'] = Queue(maxsize=0)
    models=info['device_id']
    try:
        model_dict['save_img'] = info['save_img']
    except:
        model_dict['save_img'] = False
    print("post cls model for {} save image:{}".format(model_dict['defect'],model_dict['save_img']))

    for m in models.keys():
        for i in range(int(models[m])):
            device_id = m
            device1= torch.device(f'cuda:{device_id}')
            if info['backbone'] in BACKBONE_DICT.keys():
                model = BACKBONE_DICT[info['backbone']](info['num_classes'])
                model.load_state_dict(torch.load(info['weight_path']), False)
                model_dict['transform'] = transforms.Compose([
                        transforms.Resize((info['img_size'], info['img_size']),
                                          interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
            else:
                model_dict['transform'] = transforms.Compose([
                        transforms.Resize((info['img_size'],info['img_size'])),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
                        ])
                model = torch.load(info['weight_path'])

            model.to(device1)
            model.eval()
            model_dict['model'].put([device1,model])
            print("loading cls model {}".format(info['weight_path']))

    return model_dict



def inference(det_models, cls_models, post_cls_models, image,image_enhance,photo_id,image_name):
    '''
    :param det_models: 检测模型字典list
    :param cls_models: 分类模型字典list
    :param image: numpy ndarray图片
    :param photo_id: 图片id号
    :return: 单张图片的预测和分类结果list
    '''
    singles = []
    # 取图时候对所有料判为NG
    # singles.append(SingleInferenceReply(class_name='huashang', xmin=100,
    #                      ymin=100, bb_width=100, bb_height=100, score=0.9))
    # return singles
    # Assign model according to photo_id
    for det_model in det_models:
        if photo_id in det_model['photo_ids']:
            enhance=det_model['enhance']
            scale_h=1.0
            scale_w=1.0
            if enhance:
                scale_w=image.shape[1]/1024
                scale_h=image.shape[0]/1024
                singles.extend(detect(det_model,post_cls_models, image_enhance, image_name,scale_w,scale_h))
            elif det_model['resize']:
                scale_w=image.shape[1]/det_model['resize']
                scale_h=image.shape[0]/det_model['resize']
                singles.extend(detect(det_model, post_cls_models,image, image_name,scale_w,scale_h))
            else:
                scale_w=1.0
                scale_h=1.0
                singles.extend(detect(det_model,post_cls_models, image, image_name,scale_w,scale_h))
     
    for cls_model in cls_models:
        if photo_id in cls_model['photo_ids']:
            enhance=cls_model['enhance']
            if enhance:
                singles.extend(classify(cls_model, image_enhance, image_name))
            else:
                singles.extend(classify(cls_model, image, image_name))
    return singles


def detect(det_model,post_cls_models, image, image_name,scale_w=1.0,scale_h=1.0):
  
    model1 = det_model['model'].get()
    enhance=det_model['enhance']
    analyze_defects = det_model['analyze_defects']
    device = model1[0]
    real_model= model1[1]
    # 预处理图片
    if enhance:
        image1=copy.deepcopy(image)
    elif det_model['resize']:
        image1=cv2.resize(image,(det_model['resize'],det_model['resize']))
    else:
        image1=image
    img = letterbox(image1, (math.ceil(image1.shape[0] / 32) * 32, math.ceil(image1.shape[1] / 32) * 32), stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half()  # uint8 to fp16/32

    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # 等待模型空闲
    #print(device)
    #print(img)
    #print(next(real_model.parameters()).device)
    # 开始yolov5预测
    pred = real_model(img, augment=False)[0]
    # NMS
    pred = non_max_suppression(pred, det_model['conf_thres'], det_model['iou_thres'], classes=None, agnostic=False)
    results = {'post_process': [], 'final_result': [],  'after_post': []}
    post_defect=[]
    for m in post_cls_models:
        post_defect.extend(m['defect'])

    #print(post_defect)
    #结果分类，进行预处理，切小图
    for i, det in enumerate(pred):
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image1.shape).round()
            for *xyxy, conf, cls in reversed(det):
                class_name=det_model['class_dict'][int(cls)]
                if class_name in post_defect:
                    #截图
                    img = image[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    # img = cv2.resize(crop_img, (post_cls_models[0]['img_size'], post_cls_models[0]['img_size']))
                    # img=img_crop(image,int((xyxy[2] +xyxy[0])*scale_w*0.5),int((xyxy[1] +xyxy[3])*scale_h*0.5),post_cls_models[0]['img_size'])
                    #保存到results
                    results['post_process'].append({'image_name':image_name,'class_name':det_model['class_dict'][int(cls)], 'xmin':float(xyxy[0]*scale_w),
                                              'ymin':float(xyxy[1]*scale_h), 'bb_width':float((xyxy[2] - xyxy[0])*scale_w),
                                              'bb_height':float((xyxy[3] - xyxy[1])*scale_h), 'score':float(conf),'image':img.copy()})
                else:
                    results['final_result'].append({'class_name':det_model['class_dict'][int(cls)], 'xmin':float(xyxy[0]*scale_w),
                                              'ymin':float(xyxy[1]*scale_h), 'bb_width':float((xyxy[2] - xyxy[0])*scale_w),
                                              'bb_height':float((xyxy[3] - xyxy[1])*scale_h), 'score':float(conf)})
    det_model['model'].put(model1)
    #开始处理需要二分类的图
    print("there are {} in post processing".format(len(results['post_process'])))
    thread_pool = ThreadPoolExecutor(max_workers=10)
    all_tasks=[]

    for r1 in results['post_process']:
        all_tasks.append(thread_pool.submit(post_cls, post_cls_models,r1))
    wait(all_tasks)

    if results['post_process']: print("there are {} in after post".format(len(results['post_process'])))
    results['final_result'].extend(results['post_process'])


    #开始组装成grpc result
    if results['final_result']: print("there are {} in final result".format(len(results['final_result'])))
    replys=[]

    # 开始处理生成后处理变量
    thread_pool_analyze = ThreadPoolExecutor(max_workers=10)
    all_tasks_analyze = []
    for r1 in results['final_result']:
        if r1['class_name'] in analyze_defects:
            r1['image_np'] = image.copy()
            all_tasks_analyze.append(thread_pool_analyze.submit(mask2feature, r1))
    wait(all_tasks_analyze)

    for det in results['final_result']:
        if det_model['filter'] is not None and _filter(det_model['filter'], det):
            det['image']=""
            det['image_np']=""
            logging.debug('{} detection {} is filtered!'.format(image_name,det))
            continue

        if 'image_np' in det.keys():
            single = SingleInferenceReply(class_name=det['class_name'], xmin=det['xmin'],
                                          ymin=det['ymin'], bb_width=det['bb_width'],
                                          bb_height=det['bb_height'], score=det['score'],
                                          length=det['length'],
                                          width=det['width'],
                                          pixel_area=det['pixel_area'],
                                          gradients=det['gradients'],
                                          contrast=det['contrast'],
                                          brightness=det['brightness'],
                                          max20brightness=det['max20brightness'],
                                          min20brightness=det['min20brightness'],
                                          points=det['points'],
                                          )
        else:
            single = SingleInferenceReply(class_name=det['class_name'], xmin=det['xmin'],
                                              ymin=det['ymin'], bb_width=det['bb_width'],
                                              bb_height=det['bb_height'], score=det['score'])

        
        logging.info(
            f'Image {image_name} detection: {single.class_name}, x: {single.xmin}, y: {single.ymin}, w: {single.bb_width}, '
            f'h: {single.bb_height}, score: {single.score}')
        print(
            f'Image {image_name} detection: {single.class_name}, x: {single.xmin}, y: {single.ymin}, w: {single.bb_width}, '
            f'h: {single.bb_height}, score: {single.score}')
        replys.append(single)
    return replys

def post_cls(post_cls_models,r):
    #调用后续的二分类模型来更改标签
    for post_cls_model in post_cls_models:
        if r['class_name'] in post_cls_model['defect']:
            logging.debug("#1 {} 's {} classifing".format(r['image_name'],r['class_name']))
            model1 = post_cls_model['model'].get()
            model=model1[1]
            device = model1[0]
            try:
                transform = post_cls_model['transform']
                logging.debug("#2 {} 's {} is loading image".format(r['image_name'],r['class_name']))
                image1 = Image.fromarray(r['image'])
                img = transform(image1).unsqueeze(dim=0)
                img = img.to(device)
                output = model(img)
                logging.debug("#3 {} 's {} is  inferenced".format(r['image_name'],r['class_name']))
                output = functional.softmax(output, dim=1)
                _, predict = output.topk(1, dim=1, largest=True)

                # class_id = 1 if float(output[0][1]) >= 0.60 else 0
                # score = float(output[0][1])
                class_id = int(predict)
                score = float(output[0][class_id])

                logging.debug("#4 {} 's {} is  inferenced, prediction is {}".format(r['image_name'],r['class_name'],class_id))
                class_name = post_cls_model['class_dict'][class_id]
                logging.debug("#5 {} 's {} classified to {}, score is {}".format(r['image_name'],r['class_name'],class_name,score))
                post_cls_model['model'].put(model1)
                t=time.time()*1000
                r['class_name']=class_name
        
                logging.debug("image {} class {} is in cls processing,save image: {}".format(r['image_name'], class_name, post_cls_model['save_img']))
                if post_cls_model['save_img'] is True:
                    jpg_name = r['image_name'].split('.')[0]+'-'+str(round(t))+".jpg"
                    taskid = r['image_name'][0:4]

                    save_crop_dir = os.path.join(project_dir, "inference", "1_output_resnet18", taskid, class_name)
                    if not os.path.exists(save_crop_dir):
                        os.makedirs(save_crop_dir)
                    cv2.imwrite(os.path.join(save_crop_dir, jpg_name), r['image'], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            except Exception as e:
                logging.error('{} fails in post inference , due to {}'.format(r['image_name'],e))

def _filter(filter, single):
    '''
    后处理
    输入:规则列表，单条模型预测记录
    输出：是否过滤掉
    '''
    result=False
    for vulnerability in filter['rules']:
        try:
            rule = rule_engine.Rule(vulnerability['rule'])
            #print(rule)
        except rule_engine.RuleSyntaxError as error:
            print(error.message)
        result=rule.matches(single)
        #print(result)
    return result


def classify(cls_model, image, image_name):
    results = []
    device = cls_model['device']
    transform = cls_model['transform']
    enhance=cls_model['enhance']
    resize=cls_model['resize']
    if resize:
        #image_enhance = enhance_resize(image, 100, 1024)
        image = cv2.resize(image, (resize, resize))
    image1 = Image.fromarray(image)
    # print(image)
    img = transform(image1).unsqueeze(dim=0)
    img = img.to(device)
    #img = img.cuda()
    # 等待模型空闲
    #if resize: img=img.half()
    model = cls_model['model'].get()
    output = model(img)
    output = functional.softmax(output, dim=1)
    _, predict = output.topk(1, dim=1, largest=True)
    class_id = int(predict)
    score = float(output[0][class_id])
    #print(score)
    class_name = cls_model['class_dict'][class_id]
    if class_name != cls_model['class_ok_name']:
        class_det={'class_name' : class_name, 'xmin' : 100, 'ymin' : 100, 'bb_width' : 200, 'bb_height' : 200, 'score' : score}
        single = SingleInferenceReply(class_name=class_name, xmin=100, ymin=100, bb_width=200, bb_height=200, score=score)
        if cls_model['filter'] is not None and _filter(cls_model['filter'], class_det):
            print(f'image_name {image_name} Filtered:  cls: {single.class_name}, x: {single.xmin}, y: {single.ymin}, w: {single.bb_width}, '
                f'h: {single.bb_height}, score: {single.score}') 
        else:
            print(f'image_name {image_name} cls: {single.class_name}, x: {single.xmin}, y: {single.ymin}, w: {single.bb_width}, '
                f'h: {single.bb_height}, score: {single.score}') 
            results.append(single)
        logging.info(f'image: {image_name} cls: {single.class_name}, x: {single.xmin}, y: {single.ymin}, w: {single.bb_width}, '
                     f'h: {single.bb_height}, score: {single.score}')

    # 释放模型
    cls_model['model'].put(model)
    return results


if __name__ == '__main__':
    # 加载固定配置文件libcom_config.yaml
    with open('../configs/libcom_config2.yaml', 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f.read())
    # 打印yaml config中的内容
    standard_print = pprint.PrettyPrinter(indent=2)
    standard_print.pprint(config_dict)
    if not os.path.exists('../log'):
        os.mkdir('../log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='../log/model_server.log',
                        filemode='w')
    # 启动服务器
    server_start(config_dict)
