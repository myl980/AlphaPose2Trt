项目简介：
将AlphaPose 的FastPose-RestNet50模型转为onnx再转为tensorrt模型

主要环境：
python 3.6
cuda 10.2
tensorrt 7.1
torch 1.2
numpy 1.17

安装：
 1. 下载AlphaPose:  git clone https://github.com/MVIG-SJTU/AlphaPose.git
 2.  安装AlphaPose 安装说明进行部署：  https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md
 3.  将此项目文件拷贝到AlphaPose文件夹下
 4.  修改AlphaPose/alphapose/utils/metrics.py 第121行：return info_str['AP'] → return info_str

文件说明：
1. fastpose2onnx.py 转固定批次onnx
2. fastpose2onnxDynamic.py 转动态批次onnx
3. onnx2DynamicTrt.py 将onnx转为dynamic的tensorrt模型：修改mode参数，可以转为不同精度的模型
4. validate_timeAndAP 计算推理平均时间和AP

validate_timeAndAP使用方式：
1.测试原始文件不同batch_size的效果：  python validate_timeAndAP  --batch 10
2.测试trt模型不同batch_size的效果： python validate_timeAndAp --engine_file_path trtfiles/fastpose_ret50_fp32_dynamic.trtengine --batch 10
