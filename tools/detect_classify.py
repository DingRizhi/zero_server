
import torch
import os,sys 
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir) 
import glob
import cv2
import argparse
import json
from shutil import copyfile
from tools_utils.utils import adaptivehistogram_enhance
from wlib.libcom_cls import basic_classifier_mobilenetv3,basic_classifier_resnet50,\
    basic_classifier_densenet121,basic_classifier_efficientnet_b0,basic_classifier_mobilenetv2,basic_classifier_mobilenetv3,basic_classifier_mobilenetv3_small
from PIL import Image
from torch.nn import functional
from torchvision import transforms

BACKBONE_DICT = {'s':basic_classifier_mobilenetv3_small,
                'm':basic_classifier_mobilenetv3,
                'resnet50':basic_classifier_resnet50,
                'mobilenetv2':basic_classifier_mobilenetv2,
                'densenet':basic_classifier_densenet121,
                'l':basic_classifier_efficientnet_b0
                }
def detect_classify(weight_path,img_path,img_size,num_classes,backbone,is_enhance,is_pick,target_path):

    device = 'cuda:0'
    try:
        model=BACKBONE_DICT[backbone](num_classes)
    except:
        print("Unknown backbone!!!")
    model.load_state_dict(torch.load(weight_path), False)
    img_items = glob.glob(os.path.join(img_path,"*.jpg"))
    #img_items = glob.glob(os.path.join(img_path,"*-1[4,6].jpg"))
    class_idx={}
    result={}
    for i in range(10):
        class_idx[i]='class_'+str(i)
        result[class_idx[i]]=0
    size = (img_size, img_size)

    for img_item in img_items:
        img = cv2.imread(img_item)
        (p,fn)=os.path.split(img_item)
        if is_enhance:
            flag, img = adaptivehistogram_enhance(
                img, clipLimit=100.0, tileGridSize=(8, 8))
        #img=cv2.resize(img,(512,512))
        model.to(device)
        model.eval()  # half().
        #normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
#                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        normalize=transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        transform = transforms.Compose(
            [transforms.Resize(size), transforms.ToTensor(),normalize])  # normalize
        image = Image.fromarray(img)
        img = transform(image).unsqueeze(dim=0)
        #print(img.shape)
        # print(img)
        img = img.to(device)
        # img = img.half()
        output = model(img)
        #print(output)
        output = functional.softmax(output, dim=1)
        #print(output)
        _, predict = output.topk(1, dim=1, largest=True)
        class_id = int(predict)
        #print(class_id)
        print(fn+"\t"+str(class_idx[class_id]))
        if is_pick:
            output_path=os.sep.join([target_path,class_idx[class_id]])
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            filename=os.path.basename(img_item)
            copyfile(img_item,os.sep.join([output_path,filename]))
        result[str(class_idx[class_id])]=result[str(class_idx[class_id])]+1
    print(result)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='detect_classify.py')
    parser.add_argument('--weights', type=str, default='../weights/bg_huashang_v1.pt', help='model.pt path(s)')
    parser.add_argument('--img_size', type=int, default=224, help='inference size (pixels)')
    parser.add_argument('--num_classes', type=int, default=3, help='inference num classes')
    parser.add_argument('--backbone', default='s', help='path to test')
    parser.add_argument('--test_path', default='../test/testimg/', help='path to test')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--pick', action='store_true', help='pick inference result')
    parser.add_argument('--target_path', default='./test/ok', help='inference target path')
    opt = parser.parse_args()
    print(opt)
    detect_classify(opt.weights,
            opt.test_path,
            opt.img_size,
            opt.num_classes,
            opt.backbone,
            opt.augment,
            opt.pick,
            opt.target_path
        )

