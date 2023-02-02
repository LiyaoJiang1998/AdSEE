from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import torch
import numpy as np
from PIL import Image
import cv2

import os
from tqdm import tqdm

# # download the checkpoint from model zoo and put it in `checkpoints/`
# config_file = '../configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py'
# checkpoint_file = '../checkpoints/DECOUPLED_SOLO_R50_3x.pth'

# # solo v1
# config_file = '../configs/solo/decoupled_solo_r101_fpn_8gpu_3x.py'
# checkpoint_file = '../checkpoints/DECOUPLED_SOLO_R101_3x.pth'

# solo v2
config_file = '../configs/solov2/solov2_r101_dcn_fpn_8gpu_3x.py'
checkpoint_file = '../checkpoints/SOLOv2_R101_DCN_3x.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

directory = "input_images/"
out_directory = "output_images/"
    
for filename in tqdm(os.listdir(directory)):
    if filename.endswith(".jpg"):
        img = os.path.join(directory, filename) 
        
        # test on image
        result = inference_detector(model, img)
        
        # save the result image
        output_img_array = show_result_ins(img, result, model.CLASSES, score_thr=0.25)
        output_img = Image.fromarray(cv2.cvtColor(output_img_array, cv2.COLOR_BGR2RGB), 'RGB')
        output_img.save(os.path.join(out_directory, filename.replace(".jpg", "_out.jpg")))
        