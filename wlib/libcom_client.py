import sys
import argparse
sys.path.append('../')
import json
import os
import time
from concurrent.futures._base import wait
from concurrent.futures.thread import ThreadPoolExecutor
import glob
import cv2
import grpc

from wlib.libcom_pb2 import InferenceRequest
from wlib.libcom_pb2_grpc import InferenceStub
from utils.plots import plot_one_box


def test_grpc_client(addr, img_folder_path, target_folder_path):
    if not os.path.exists(target_folder_path):
        os.mkdir(target_folder_path)
        #os.mkdir(os.path.join(target_folder_path, 'OK'))
    if not os.path.exists(os.sep.join([target_folder_path, 'NG'])):
        os.mkdir(os.sep.join([target_folder_path, 'NG']))
    channel = grpc.insecure_channel(addr)
    stub = InferenceStub(channel)
    thread_pool = ThreadPoolExecutor(max_workers=10)
    t0 = time.time()
    all_tasks = []
    num_imgs = 0
    result={'ok':0,'ng':0}
    imgs=glob.glob(os.sep.join([img_folder_path,"*.jpg"]))
    for img_file_path in imgs:
        num_imgs += 1
        img_file=os.path.basename(img_file_path)
        img = cv2.imread(img_file_path)
        try:
            img_name_list=img_file.split('.')[0].split('-')
        except Exception as e:
            print(e,"photo_id is set to 1")
            img_name_list[0,0,1]
        photo_id = int(img_name_list[2])
        #print('photo_name {}'.format(img_name_list))
        request = InferenceRequest(photo_id=photo_id, channel_id=int(img_name_list[0]), product_id=int(img_name_list[1]),width=img.shape[1], height=img.shape[0],
                                   encoded_image=img.tobytes())
        all_tasks.append(thread_pool.submit(run, stub, request, photo_id, img_file, img, target_folder_path, result))
    wait(all_tasks)
    print(f'Average total time {(time.time() - t0) / num_imgs} second per image')
    print('There are {} ng and {} ok'.format(result['ng'],result['ok']))


def run(stub, request, photo_id, img_file, img, target_folder_path,result):
    t0 = time.time()
    reply = stub.Inference(request)
    print(f'Image {photo_id} Inference Finished in {time.time() - t0} s')
    singles = reply.singleInferenceReply
    print(singles)
    if len(singles) == 0:
        result['ok']+=1
        #cv2.imwrite(os.path.join(target_folder_path, 'OK', img_file), img)
        pass
    else:
        result['ng']+=1
        print("{} is ng".format(img_file))
        json_dict = {"version": "4.2.10", "flags": {}, "shapes": [], "imagePath": img_file,
                     "imageData": None,
                     "imageHeight": img.shape[0],
                     "imageWidth": img.shape[1]}
        if True:
            for single in singles:
                xyxy = (single.xmin, single.ymin, single.xmin + single.bb_width,
                        single.ymin + single.bb_height) if single.xmin != 1 else (100, 100, 101, 101)
                plot_one_box(xyxy, img, None, single.class_name + '_' + str(single.score), line_thickness=1)
                shape = {"label": single.class_name, "points": [[single.xmin, single.ymin],
                                                                [single.xmin + single.bb_width,
                                                                 single.ymin + single.bb_height]],
                         "group_id": {}, "shape_type": "rectangle","score":single.score,
                         "flags": {}}
                json_dict['shapes'].append(shape)
        cv2.imwrite(os.sep.join([target_folder_path, 'NG', img_file]), img)
        json_fp = open(os.sep.join([target_folder_path, 'NG', img_file.replace('.jpg', '.json')]), 'w')
        #print(os.path.join(target_folder_path, img_file.replace('.jpg', '.json')))
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
        #print(json_dict)
        json_fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='libcom_client.py')
    # parser.add_argument('--img_path', default="../test/testimg", help='path to test')
    # parser.add_argument('--target', default="../inference/output/", help='path to save')
    parser.add_argument('--img_path', default="/data/fc_watch/1/original", help='path to test')
    parser.add_argument('--target', default="../inference/1_output_resnet18/", help='path to save')
    opt = parser.parse_args()
    print(opt)
    test_grpc_client(addr='127.0.0.1:9099',img_folder_path=opt.img_path,target_folder_path=opt.target)
