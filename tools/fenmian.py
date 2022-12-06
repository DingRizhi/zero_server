import glob
import os
from shutil import copyfile
from tools_utils import *

def fenmian(from_list:list,new_dic:str,selected:list):
    if os.path.exists(new_dic):
        #shutil.rmtree(new_dic)
        print("Target path exist")
    if not os.path.exists(new_dic):
        os.makedirs(new_dic)
    for dir in from_list:
        pics = glob.glob('{}/*'.format(dir))
        for pic in pics:
            filename=os.path.basename(pic)
            try:
                weiyi_file_name=filename[0:12]
                pic_id=int(weiyi_file_name[10:12])
            except:
                weiyi_file_name=filename
                pic_id=1
                continue
            #pic_id = pic.split('/')[-1].split('.')[0].split('-')[-1]
            #pic_id = int(pic_id)
            if pic_id in selected:
                sp = os.sep.join([new_dic, filename])
                copyfile(pic,sp)
                print(pic,' was copied')

if __name__ == "__main__": 
    from_list = ['/media/adt/data1/backup/0_wr_gongjian/OK_old/A1hei2020',
                '/media/adt/data1/backup/0_wr_gongjian/OK_old/A1hei2021',
                '/media/adt/data1/backup/0_wr_gongjian/OK_old/A2-1139',
                '/media/adt/data1/backup/0_wr_gongjian/OK_old/hei-1092/original_pictures',
                '/media/adt/data1/backup/0_wr_gongjian/OK_old/yin-1083/original_pictures'
                ]


    # from_list = ['/media/adt/ZX2 1TB/0_wr_gongjian/NG_MAY/meb5月前50%的数据/origin','/media/adt/ZX2 1TB/0_wr_gongjian/NG_MAY/meb5月前50%的数据/xml']
    new_dic = '/media/adt/data1/backup/0_wr_gongjian/fenmian/dmys_ok_1105'
    selected = [13,14,15,16]
    fenmian(from_list=from_list,
            new_dic=new_dic,
            selected=selected)
