## 运行服务
cd wlib   
python -m libcom_server  
即可运行模型

## 测试服务
cd wlib  
python libcom_client.py  
发送test/testimg里面的图片确认结果:  
  
Average total time 0.1956021636724472 second per image  
There are 2 ng and 14 ok

## todo

- [x] 改造libcom_client.py，能够画框
- [ ] 改造libcom_client.py，输出json，以及结果统计表
- [ ] 改造小工具系统，可以图像化调用不同的脚本
- [ ] 改造libcom_server，实现模型状态监控
- [ ] 改造libcom_server，支持onnx推理，yolox推理
- [x] 改造rule engine系统，能够做好屏蔽
# zero_server
