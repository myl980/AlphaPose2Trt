import torch
import onnx
from alphapose.models.fastpose import FastPose
from alphapose.models import builder
from alphapose.utils.config import update_config

'''
将pth模型转为onnx的动态输入模型
'''
def saveONNX(model, filepath,c, h, w):
    #输入数据形状
    dummy_input = torch.zeros(1, c, h, w, device='cuda')
    dynamic_ax = {'input': {0:'batch_size'},
                  'output': {0:'batch_size'}}
    torch.onnx.export(model, dummy_input, filepath, opset_version=10, input_names=["input"], output_names=["output"], dynamic_axes=dynamic_ax)

cfg = update_config("configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml")
print(cfg)
pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
pose_model.load_state_dict(torch.load("pretrained_models/fast_res50_256x192.pth"))
pose_model.eval()
print(pose_model)
pose_model = pose_model.cuda()

saveONNX(pose_model, filepath="onnxfile/fastpose_ret50_dynamic.onnx", c=3, h=256, w=192)


