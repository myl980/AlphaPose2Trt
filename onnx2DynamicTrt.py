import tensorrt as trt
import os
from calibrator import MyCalibrator

# 查看版本
print(trt.__version__)

'''
仅在trt7(7.1.3.4)上可以转成功，同时onnx也必须是动态输入才能转为动态输入的trt模型
onnx 转tensorrt动态批次， 可转模型：
   fastpose_ret50_fp32_dynamic.trtengine
   
   
存在问题：这个代码转fp32和fp16还行，转int8感觉有问题(虽然能够转出来，且可以推理，但转出来的模型比fp16的还大，而且推理速度也比fp16慢一点)
'''
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def get_engine(onnx_file_path,engine_file_path):
    def build_engine():
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_batch_size = 1 #这个设1就行

            # builder.max_workspace_size = 1 << 28 # 256MiB
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  #注意，转动态批次的话，max_workspace_size是通过config设置的，而不是builder
            if mode=="fp16":
                # builder.fp16_mode = True
                config.flags = 1 << int(trt.BuilderFlag.FP16)   #注意，转动态批次的话，fp16也是是通过config设置的，而不是builder
            elif mode == "int8":
                # builder.int8_mode = True
                config.flags = 1 << int(trt.BuilderFlag.INT8)  # 注意，转动态批次的话，int8也是是通过config设置的，而不是builder

                # dynamic int8的校准批次(输入形状)一般等于profile.set_shape的opt批次，即第二个
                # 不过不传好像也没啥问题
                config.int8_calibrator = MyCalibrator(batch_size=16)


            if is_dynamic:
                print('----------设置动态输入----------')
                profile = builder.create_optimization_profile()
                #这里是仅批次是动态的，所以仅批次大小在变化，最小是1，最大是32(也就是推理是最多传32批次的数据)
                profile.set_shape(input_name, (1, 3, 256, 192), (16, 3, 256, 192), (32, 3, 256, 192))
                config.add_optimization_profile(profile)
                # config.set_calibration_profile(profile) #这一句好像没啥用

            onnx_model=open(onnx_file_path, 'rb')
            parser.parse(onnx_model.read())
            engine = builder.build_engine(network, config)

            file=open(engine_file_path, "wb")
            file.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


if __name__ == '__main__':
    is_dynamic = True
    input_name = 'input'  #netron_onnx.py 查看层的名字
    #获取trt日志记录
    TRT_LOGGER = trt.Logger()
    #转换trtengine的精度,也可以选择fp16
    #mode = 'fp32'
    mode = 'fp16'
    # mode = 'int8'
    #onnx模型路径
    onnx_model_path="onnxfile/fastpose_ret50_dynamic.onnx"
    engine_file_path="trtfiles/fastpose_ret50_{}_dynamic.trtengine".format(mode)
    #trt模型的批次，默认为1
    trt_batch=1


    get_engine(onnx_model_path,engine_file_path)
