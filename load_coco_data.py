from pycocotools.coco import COCO
import cv2
from alphapose.utils.presets import SimpleTransform
import numpy as np

root_dir = './datasets/COCO2017/'
json_path = 'annotations/person_keypoints_train2017.json'
img_dir = 'train2017/'

'''fastPose 前处理'''
class poseDataset():
    def __init__(self):
        self.CLASSES = ['person']
        self.EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.joint_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        self.lower_body_ids = (11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25)
        self.num_joint = 17


pose_dataset = poseDataset()
transformation = SimpleTransform(
    pose_dataset, scale_factor=0,
    input_size=[256, 192],
    output_size=[64, 48],
    rot=0, sigma=2,
    train=False, add_dpg=False, gpu_device='cuda:0')


def load_coco_data(root_dir = root_dir, json_path = json_path, img_dir = img_dir):
    coco_data = COCO(root_dir + json_path)
    img_ids = coco_data.getImgIds()
    imgs = coco_data.imgs
    anns = coco_data.imgToAnns
    num = 0
    inputs = []
    for id in img_ids:
        num += 1
        img_name = imgs[id]["file_name"]
        print('img_name: ', img_name)
        img = cv2.imread(root_dir + img_dir + img_name)
        for ann in anns[id]:  #图片信息
            if ann['num_keypoints'] == 0:
                continue
            bbox = ann['bbox']
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[0] + bbox[2])
            y2 = int(bbox[1] + bbox[3])
            inp, cropped_box = transformation.test_transform(img, [x1, y1, x2, y2])
            # print('inp:', inp)
            inputs.append(np.array(inp))
        if num >= 1000:
            break
    inputs = np.array(inputs)
    print('inputs.shape:', inputs.shape)

    return np.ascontiguousarray(inputs.astype(np.float32))



#
# load_coco_data(root_dir, json_path, img_dir)












'''coco获取数据信息测试'''
# root_dir = '/e/myl/datasets/COCO2017/'
# json_path = 'annotations/person_keypoints_train2017.json'
# img_dir = 'train2017/'
# coco_train = COCO(root_dir + json_path)
# img_ids = coco_train.getImgIds()
# imgs = coco_train.imgs
# anns = coco_train.imgToAnns
# for id in img_ids:
#     img_name = imgs[id]["file_name"]
#     print('img_name: ',img_name)  #图片名
#     img = cv2.imread(root_dir + img_dir + img_name)
#     for a in anns[id]:  #图片信息
#         print('a.keys(): ', a.keys())
#         print("a['num_keypoints']: ", a['num_keypoints'])
#         if a['num_keypoints'] == 0:
#             continue
#         print("a['keypoints']: ", a['keypoints'])
#         bbox = a['bbox']
#         print('bbox: ',bbox)
#         x1 = int(bbox[0])
#         y1 = int(bbox[1])
#         x2 = int(bbox[0] + bbox[2])
#         y2 = int(bbox[1] + bbox[3])
#
#         cv2.rectangle(img, (x1, y1), (x2, y2),(255,0,0))
#     cv2.imshow('img', img)
#     cv2.waitKey(0)

