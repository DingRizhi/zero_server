import dearpygui.dearpygui as dpg

from fenmian import fenmian
from pick_defect import pick_defect
from trans_img import trans
from gpu_status import Monitor
from concurrent.futures._base import wait
from concurrent.futures.thread import ThreadPoolExecutor
import sys
import argparse
sys.path.append('../')
from detect_classify import detect_classify
from wlib.libcom_client import yolov5_grpc_client
monitor = Monitor(0.5)
#monitor.start()
gpu_nums=monitor.get_gpu_num()
x_data=[]
cosdatax = []
cosdatay = []


def update_series():
    if len(cosdatax)>10:
        x_data=[i for i in range(10)]
        dpg.set_value('series_tag', [x_data, cosdatay[-11:]])
        dpg.set_value('series_tag2', [x_data, cosdatax[-11:]])
        dpg.set_item_label('series_tag', "GPU used")
        dpg.set_item_label('series_tag2', "GPU Mem used")



dpg.create_context()
def button_callback(sender, app_data):
    print(f"sender is: {sender}")
    print(f"app_data is: {app_data}")
    from_list=[dpg.get_value("source1"),dpg.get_value("source2"),dpg.get_value("source3"),dpg.get_value("source4")]
    new_dic=dpg.get_value("output_path")
    selected=list(map(lambda x:int(x),dpg.get_value("selected_pic").split(",")))
    fenmian(from_list=from_list,
            new_dic=new_dic,
            selected=selected)
def button_pd_callback(sender, app_data):
    print(f"sender is: {sender}")
    print(f"app_data is: {app_data}")
    from_list=[dpg.get_value("pd_json_file_folder"),dpg.get_value("pd_pic_file_folder"),dpg.get_value("pd_target_file_folder"),int(dpg.get_value("pd_img_size"))]
    if ',' in dpg.get_value("pd_defects"):
        selected=dpg.get_value("pd_defects").split(',')
    else:
        selected=[dpg.get_value("pd_defects")]

    #selected=list(map(lambda x:int(x),dpg.get_value("pd_defects").split(",")))
    pick_defect(from_list[0],
            from_list[1],
            from_list[2],
            selected,
            from_list[3],
        )
def button_enhance_callback(sender, app_data):
    print(f"sender is: {sender}")
    print(f"app_data is: {app_data}")
    from_list=[dpg.get_value("enhance_origin_folder"),dpg.get_value("enhance_target_folder"),int(dpg.get_value("enhance_img_size"))]
  
    #selected=list(map(lambda x:int(x),dpg.get_value("pd_defects").split(",")))
    trans(from_list[0],
            from_list[1],
            from_list[2],
        )
def button_test_callback(sender, app_data):
    print(f"sender is: {sender}")
    print(f"app_data is: {app_data}")
    from_list=[dpg.get_value("test_origin_folder"),dpg.get_value("test_save_folder"),dpg.get_value("test_addr")]
  
    #selected=list(map(lambda x:int(x),dpg.get_value("pd_defects").split(",")))
    thread_pool = ThreadPoolExecutor(max_workers=10)
    all_tasks=[]
    all_tasks.append(thread_pool.submit(yolov5_grpc_client,from_list[2], from_list[0],from_list[1]))
    wait(all_tasks)

def button_cls_callback(sender, app_data):
    print(f"sender is: {sender}")
    print(f"app_data is: {app_data}")
    from_list=(dpg.get_value("cls_weight"),
     dpg.get_value("cls_origin_folder"),
     dpg.get_value("cls_img_size"),
     dpg.get_value("cls_num_class"),
     dpg.get_value("cls_backbone"),
     dpg.get_value("cls_enhance"),
     dpg.get_value("cls_save_img"),
     dpg.get_value("cls_save_folder"))
    #selected=list(map(lambda x:int(x),dpg.get_value("pd_defects").split(",")))
    print(from_list)
    detect_classify(*from_list)
    #thread_pool = ThreadPoolExecutor(max_workers=10)
    #all_tasks=[]
    #all_tasks.append(thread_pool.submit(detect_classify,from_list))
    #wait(all_tasks)

with dpg.value_registry():
    #传入分面程序的变量
    dpg.add_string_value(default_value="/media/adt/8234A94B34A942D1/nfsdata/a/OK/A1hei2020", tag="source1")
    dpg.add_string_value(default_value="/media/adt/8234A94B34A942D1/nfsdata/a/OK/A1hei2021", tag="source2")
    dpg.add_string_value(default_value="/media/adt/8234A94B34A942D1/nfsdata/a/OK/A1hei-2203/original_pictures", tag="source3")
    dpg.add_string_value(default_value="/media/adt/8234A94B34A942D1/nfsdata/a/OK/A3-773-yin/original_pictures", tag="source4")
    dpg.add_string_value(default_value="5,6,7,8,9,10,11,12", tag="selected_pic")
    dpg.add_string_value(default_value="/data/meboffline/lupinus_device/gk_cmp_ok_1111", tag="output_path")
    #传入挑选缺陷程序的变量
    dpg.add_string_value(default_value="/home/adt/data/Pictures/0100/original_pictures", tag="pd_json_file_folder")
    dpg.add_string_value(default_value="/home/adt/data/Pictures/0100/original_pictures", tag="pd_pic_file_folder")
    dpg.add_string_value(default_value="/home/adt/data/new/", tag="pd_target_file_folder")
    dpg.add_string_value(default_value="pengshang,huashang", tag="pd_defects")
    dpg.add_string_value(default_value=224, tag="pd_img_size")

    #传入增强变换的变量
    dpg.add_string_value(default_value="/home/adt/data/Pictures/0100/original_pictures", tag="enhance_origin_folder")
    dpg.add_string_value(default_value="/home/adt/data/Pictures/new", tag="enhance_target_folder")
    dpg.add_string_value(default_value=1024, tag="enhance_img_size")

    #Testdata
    dpg.add_string_value(default_value="../test/testimg/", tag="test_origin_folder")
    dpg.add_string_value(default_value="../../inference/output", tag="test_save_folder")
    dpg.add_string_value(default_value='127.0.0.1:9099', tag="test_addr")


with dpg.window(label="Split Picture by ID",pos=(0,0),width=800, collapsed =True):
    dpg.add_text("I will Split the pictures to the folder")
    
    dpg.add_input_text(label="Source1", source="source1")
    dpg.add_input_text(label="Source2",  source="source2")
    dpg.add_input_text(label="Source3", source="source3")
    dpg.add_input_text(label="Source4", source="source4")
    dpg.add_input_text(label="Select", source="selected_pic")
    dpg.add_input_text(label="OUT",  source="output_path")
    dpg.add_button(label="copy to path!!!",callback=button_callback)


with dpg.window(label="Trans enhance dataset", pos=(0,25),width=800, collapsed =True):
    dpg.add_text("I will enhance from dataset to the folder")
    dpg.add_input_text(label="Input file folder", source="enhance_origin_folder")
    dpg.add_input_text(label="Output folder", source="enhance_target_folder")
    dpg.add_input_text(label="img_size",  source="enhance_img_size")
    dpg.add_button(label="enhance!!!!",callback=button_enhance_callback)


with dpg.window(label="pick defect from dataset", pos=(0,50),width=800, collapsed =True):
    dpg.add_text("I will pick defect from dataset to the folder")
    dpg.add_input_text(label="Json file folder", source="pd_json_file_folder")
    dpg.add_input_text(label="Pic file folder",  source="pd_pic_file_folder")
    dpg.add_input_text(label="Output folder", source="pd_target_file_folder")
    dpg.add_input_text(label="Defects: ex, 'pengshang,huashang'",  source="pd_defects")
    dpg.add_input_text(label="img_size",  source="pd_img_size")
    dpg.add_button(label="pick!!!!",callback=button_pd_callback)

with dpg.window(label="GPU", tag="win", pos=(0,75),width=800, collapsed =True):
    # create plot
    with dpg.group(horizontal=True):
        with dpg.plot(label="GPU load (%)", height=250, width=350):
            # optionally create legend
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="time",  tag="x_axis")
            dpg.set_axis_limits(dpg.last_item(), 0.0, 10.0)
            dpg.add_plot_axis(dpg.mvYAxis, label="GPU Load", tag="y_axis")
            dpg.set_axis_limits(dpg.last_item(), 0.0, 100.0)
            # series belong to a y axis
            dpg.add_line_series(x_data, cosdatay, label="GPU load", parent="y_axis", tag="series_tag")
        
        with dpg.plot(label="GPU memory load (%)", height=250, width=350):
            # optionally create legend
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="time series",  tag="x_axis2")
            dpg.set_axis_limits(dpg.last_item(), 0.0, 10.0)
            dpg.add_plot_axis(dpg.mvYAxis, label="GPU Memery", tag="y_axis2")
            dpg.set_axis_limits(dpg.last_item(), 0.0, 100.0)
            # series belong to a y axis
            dpg.add_line_series(x_data, cosdatax, label="GPU Memery", parent="y_axis2", tag="series_tag2")

with dpg.window(label="Test dataset", pos=(0,100),width=800, collapsed =True):
    dpg.add_text("I will test dataset to save")
    dpg.add_input_text(label="Inference Input folder", source="test_origin_folder")
    dpg.add_input_text(label="Inference Output folder", source="test_save_folder")
    dpg.add_input_text(label="Server Address",  source="test_addr")
    dpg.add_button(label="test!!!!",callback=button_test_callback)


with dpg.window(label="cls model test", pos=(0,125),width=800, collapsed =True):
    dpg.add_text("I will test cls model")
    dpg.add_input_text(label="Classify weights path", default_value="../weights/bg_huashang_v1.pt",tag="cls_weight")
    dpg.add_input_text(label="Inference Input folder", default_value="../test/testimg/",tag="cls_origin_folder")
    dpg.add_input_text(label="Inference Output folder", default_value="../../inference/output", tag="cls_save_folder")
    dpg.add_combo(label="select model",items=['s','m','l'], tag="cls_backbone")
    dpg.add_input_int(label="image size", default_value=224,min_value=32, step=32,max_value=9999, tag="cls_img_size")
    dpg.add_input_int(label="class nums", default_value=2,min_value=2, step=1,max_value=10, tag="cls_num_class")
    dpg.add_checkbox(label="Save image",default_value=True,  tag="cls_save_img")
    dpg.add_checkbox(label="Enhance image",default_value=False,  tag="cls_enhance")
    dpg.add_button(label="classify!!!!",callback=button_cls_callback)

dpg.create_viewport(title='Weiyi CV Toolbox', width=1024, height=768)
dpg.setup_dearpygui()
dpg.show_viewport()
jump_step=0
count=0
while dpg.is_dearpygui_running():
    # insert here any code you would like to run in the render loop
    # you can manually stop by using stop_dearpygui()
    
    if jump_step==5:
        results=monitor.get_result()
        cosdatax.append(results[0].memoryUtil*100)
        cosdatay.append(results[0].load*100)
        update_series()
        count+=1
        jump_step=0
        continue
    jump_step+=1
    dpg.render_dearpygui_frame()
dpg.start_dearpygui()
monitor.stop()
dpg.destroy_context()
