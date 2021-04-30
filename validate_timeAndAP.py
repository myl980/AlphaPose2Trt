"""Validation script."""
import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm
import time
import tensorrt as trt
import trt_common
import pycuda.driver as cuda

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.metrics import evaluate_mAP
from alphapose.utils.transforms import (flip, flip_heatmap,heatmap_to_coord_simple,
                                        get_func_heatmap_to_coord)
from alphapose.utils.pPose_nms import oks_pose_nms


parser = argparse.ArgumentParser(description='AlphaPose Validate')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    default='configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
                    type=str)
parser.add_argument('--checkpoint',
                    help='checkpoint file name',
                    default='pretrained_models/fast_res50_256x192.pth',
                    type=str)
parser.add_argument('--engine_file_path',
                    help='checkpoint file name',
                    default=None,
                    type=str)
parser.add_argument('--gpus',
                    help='gpus',
                    default='0',
                    type=str)
parser.add_argument('--batch',
                    help='validation batch size',
                    default= 1,
                    type=int)
parser.add_argument('--flip-test',
                    default=False,
                    dest='flip_test',
                    help='flip test',
                    action='store_true')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")

opt = parser.parse_args()
cfg = update_config(opt.cfg)

gpus = [int(i) for i in opt.gpus.split(',')]
opt.gpus = [gpus[0]]
opt.device = torch.device("cuda:" + str(opt.gpus[0]) if opt.gpus[0] >= 0 else "cpu")

def allocate_buffers(engine, batch_size=None):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

    for binding in engine:
        dims = engine.get_binding_shape(binding)
        # print('buff--dims:', dims)
        if dims[0] == -1:
            assert (batch_size is not None)
            dims[0] = batch_size
        size = trt.volume(dims) * engine.max_batch_size  # The maximum batch size which can be used for inference.
        # print("size:",size)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):  # Determine whether a binding is an input binding.
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def count_time():
    torch.cuda.synchronize(0)
    now_time = time.time()
    return now_time

def validate_gt( cfg, batchsize=1,engine_file_path = None, m=None,heatmap_to_coord=None):

    gt_val_dataset = builder.build_dataset(cfg.DATASET.VAL, preset_cfg=cfg.DATA_PRESET, train=False)
    eval_joints = gt_val_dataset.EVAL_JOINTS

    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batchsize, shuffle=False, num_workers=20, drop_last=True)
    kpt_json = []
    if m:
        m.eval()

    norm_type = cfg.LOSS.get('NORM_TYPE', None)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE
    average_time = 0
    pytorch_all_infer_time = 0
    trt_all_infer_time = 0
    data_num = 0

    for inps, labels, label_masks, img_ids, bboxes in tqdm(gt_val_loader, dynamic_ncols=True):
        data_num += 1
        if engine_file_path:

            # hm_data = []
            inps = inps.numpy()
            np.copyto(inputs[0].host, inps.ravel())  

            trt_infer_before_time = count_time()
            trt_outputs = trt_common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            trt_infer_after_time = count_time()

            trt_all_infer_time += trt_infer_after_time - trt_infer_before_time

            pred = trt_outputs[0].reshape(-1,17, 64, 48)
        else:
            if isinstance(inps, list):
                inps = [inp.cuda() for inp in inps]
            else:
                inps = inps.cuda()

            pytorch_infer_before_time = count_time()
            output = m(inps)
            pytorch_infer_after_time = count_time()
            pytorch_all_infer_time += pytorch_infer_after_time - pytorch_infer_before_time

            if opt.flip_test:
                if isinstance(inps, list):
                    inps_flip = [flip(inp).cuda() for inp in inps]
                else:
                    inps_flip = flip(inps).cuda()
                output_flip = flip_heatmap(m(inps_flip), gt_val_dataset.joint_pairs, shift=True)
                pred_flip = output_flip[:, eval_joints, :, :]
            else:
                output_flip = None

            pred = output
            assert pred.dim() == 4
            pred = pred[:, eval_joints, :, :]

        for i in range(pred.shape[0]):    #后处理过程
            bbox = bboxes[i].tolist()
            if engine_file_path:
                pose_coords, pose_scores = heatmap_to_coord_simple(pred[i], bbox, hm_shape=hm_size,norm_type=norm_type)
            else:
                pose_coords, pose_scores = heatmap_to_coord(
                    pred[i], bbox, hm_shape=hm_size, norm_type=norm_type)

            keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['bbox'] = bboxes[i].tolist()
            data['image_id'] = int(img_ids[i])
            data['score'] = float(np.mean(pose_scores) + np.max(pose_scores))
            data['category_id'] = 1
            data['keypoints'] = keypoints

            kpt_json.append(data)
    if engine_file_path:
        average_time = float(trt_all_infer_time/data_num)
    else:
        average_time = (pytorch_all_infer_time/data_num)

    print("average_time:", average_time)
    res_file = r"data/coco/res.json"
    with open(res_file,'w') as F:
        json.dump(kpt_json,F)
    res = evaluate_mAP(res_file,ann_type='keypoints',ann_file=os.path.join(cfg.DATASET.VAL.ROOT, cfg.DATASET.VAL.ANN))
    return res,average_time

if __name__ == "__main__":


    engine_file_path = opt.engine_file_path
    if engine_file_path:
        TRT_LOGGER = trt.Logger()
        engine_file = open(engine_file_path, 'rb')
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_file.read())
        context = engine.create_execution_context()
        inputs, outputs, bindings, stream = allocate_buffers(engine, opt.batch) 
        context.active_optimization_profile = 0
        context.set_binding_shape(0, (opt.batch, 3,256,192))

        gt,average_time = validate_gt(cfg, opt.batch, engine_file_path=engine_file_path)  
        print("AP: {}, Ap .5: {}, AP .75: {}, AP (M): {}, AP (L): {}, average_infer_time: {}".format(gt['AP'],
                                                                                                     gt['Ap .5'],
                                                                                                     gt['AP .75'],
                                                                                                     gt['AP (M)'],
              gt['AP (L)'], average_time))

    else:
        m = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

        print('Loading model from {}...'.format(opt.checkpoint))
        m.load_state_dict(torch.load(opt.checkpoint))

        m = torch.nn.DataParallel(m, device_ids=gpus).cuda()  # 构建网络
        heatmap_to_coord = get_func_heatmap_to_coord(cfg)

        with torch.no_grad():
            gt,average_time = validate_gt(cfg,opt.batch, m = m, heatmap_to_coord=heatmap_to_coord)  
            print("AP: {}, Ap .5: {}, AP .75: {}, AP (M): {}, AP (L): {}, average_infer_time: {}".format(gt['AP'], gt['Ap .5'], gt['AP .75'], gt['AP (M)'], gt['AP (L)'], average_time))
