import torch
import onnx
from alphapose.models.fastpose import FastPose
from alphapose.models import builder
from alphapose.utils.config import update_config

def saveONNX(model, filepath,c, h, w):
    #输入数据形状
    dummy_input = torch.randn(1, c, h, w, device='cuda')
    torch.onnx.export(model, dummy_input, filepath, verbose=True)

cfg = update_config("configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml")
print(cfg)
pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
pose_model.load_state_dict(torch.load("pretrained_models/fast_res50_256x192.pth"))
pose_model.eval()
print(pose_model)
pose_model = pose_model.cuda()

saveONNX(pose_model, filepath="onnxfile/fastpose_ret50.onnx", c=3, h=256, w=192)

'''
运行完成后，使用onnx-simplifier来优化hrnet模型：python -m onnxsim hrnet_ret50.onnx hrnet_ret50-sim.onnx
这样可以直接使用onnx2trt转为trtengine使用
'''

