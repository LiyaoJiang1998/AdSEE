'''
For each raw input image, preproccess using SOLO segmentation.
1. segment all people for each image
2. save a list of masks for each people
3. save the a set of class category label

Output format:
row_dict = {"idx": idx,
            "valid_image": True
            "image_name": image_name,
            "person_count":person_count,
            "person_masks":person_count,
            "category_labels":person_count,
            "category_name_labels":person_count,}
'''
import argparse
import os
import pathlib
import sys
import warnings
from tqdm import tqdm
import joblib

import torch
from PIL import Image
from IPython.display import display
import cv2
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

import pycocotools.mask as maskUtils
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


def separate_results(img, result, class_names):
    '''
    returns a list of separated instances, each instance is a dict {index, category, confidence, mask, img_array}
    '''
    separated_results = {"category": [],
                         "category_name": [], 
                         "confidence": [], 
                         "mask": []}
    
    h, w, _ = img.shape
    cur_result = result[0]
    if cur_result is None:
        return separated_results
    
    seg_label = cur_result[0] # segmentation maps
    seg_label = seg_label.cpu().numpy().astype(np.uint8)
    cate_label = cur_result[1] # category label for each map
    cate_label = cate_label.cpu().numpy()
    score = cur_result[2].cpu().numpy() # cofidence score for each map 
    
    
    
    for idx in range(0,len(cate_label)):
        cur_mask = seg_label[idx, :, :] # binary instance mask
        cur_mask = mmcv.imresize(cur_mask, (w, h))
        cur_mask = (cur_mask > 0.5).astype(np.uint8)
        if cur_mask.sum() == 0:
            continue
        cur_mask_bool = cur_mask.astype(bool)
#         separated_img_array = np.ones((h,w,3), dtype="uint8") * 119 # blank image background
#         separated_img_array = cv2.GaussianBlur(img,(31,31), 0, cv2.BORDER_DEFAULT) # blurred image background
#         separated_img_array[cur_mask_bool] = img[cur_mask_bool] # add instance to the background

        # each instance is a dict {category, category_name, confidence, mask}
        separated_results["category"].append(cate_label[idx])
        separated_results["category_name"].append(class_names[cate_label[idx]])
        separated_results["confidence"].append(score[idx])
        separated_results["mask"].append(cur_mask_bool)
            
    return separated_results

def separate_person_others(separated_results, person_thresh, other_thresh):
    person_count = 0
    person_masks = []
    category_labels = []
    category_name_labels = []
    
    for i in range(len(separated_results["category"])):
        if separated_results["category_name"][i] == "person" and separated_results["confidence"][i] >= person_thresh:
            person_count += 1
            person_masks.append(separated_results["mask"][i])
        if separated_results["category_name"][i] != "person" and separated_results["confidence"][i] >= other_thresh:
            category_labels.append(separated_results["category"][i])
            category_name_labels.append(separated_results["category_name"][i])
    
    category_labels = sorted(list(set(category_labels)))
    category_name_labels = sorted(list(set(category_name_labels)))
    
    return person_count, person_masks, category_labels, category_name_labels


def segment_single_image(img_path, model, class_names):
    try:
        result = inference_detector(model, img_path)
    except:
        return [None]
    
    return result

def segment_batched_image(img_path_list, model, class_names):
#     results = inference_detector(model, img_path_list) # Do Not Use, mmcv does not support batched inference
    results = []
    for img_path in img_path_list:
        result = segment_single_image(img_path, model, class_names)
        results.append(result)
    return results
    
def main(args):
    # build the model from a config file and a checkpoint file
#     model = init_detector(args.config_file, args.checkpoint_file, device='cuda:0')
    
    # multi-GPU support
    model = init_detector(args.config_file, args.checkpoint_file, device='cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dp = torch.nn.DataParallel(model)
    model_dp.to(device)
    model = model_dp.module
    
    # Create Ouput Folder if not exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    # Segmentation for all images in folder
    image_names = [f for f in os.listdir(args.input_path) if f.endswith(".jpg")]
    image_names = sorted(image_names, key=lambda x: int(os.path.splitext(x)[0]))
    class_names = model.CLASSES
    if args.end_index <= 0:
        args.end_index = len(image_names)
    for idx in tqdm(range(args.start_index, args.end_index, args.batch_size)):
        batch_img_list = []
        batch_img_path_list = []
        batch_row_list = []
        batch_end_index = idx+args.batch_size if (idx+args.batch_size) <= len(image_names) else len(image_names)
        
        batch_already_done = True
        for j in range(idx, batch_end_index):
            instance_save_path = os.path.join(args.output_path, "solo_"+image_names[j].replace(".jpg",".pkl"))
            if not os.path.isfile(instance_save_path):
                batch_already_done = False
        if batch_already_done:
            continue # If this batch all done before, go to next batch
        
        for j in range(idx, batch_end_index):
            instance_save_path = os.path.join(args.output_path, "solo_"+image_names[j].replace(".jpg",".pkl"))
            image_name = image_names[j]
            img_path = os.path.join(args.input_path, image_name)
            img = mmcv.imread(img_path)
            
            if img is None:
                row_dict = {"idx": image_name.replace(".jpg",""),
                        "valid_image": False,
                        "image_name": image_name,
                        "person_count":None,
                        "person_masks":None,
                        "category_labels":None,
                        "category_name_labels":None,}
                batch_row_list.append(row_dict)
            else:
                # img load success, valid_image = True
                row_dict = {"idx": image_name.replace(".jpg",""),
                            "valid_image": True,
                            "image_name": image_name,
                            "person_count":None,
                            "person_masks":None,
                            "category_labels":None,
                            "category_name_labels":None,}
                batch_row_list.append(row_dict)
                batch_img_list.append(img)
                batch_img_path_list.append(img_path)
                
#         results = segment_batched_image(batch_img_path_list, model, class_names)
        # split batch into GPU batches to fit into GPU by gpu_batch_size
        results = []
        for b in range(0, len(batch_img_path_list), args.gpu_batch_size):
            b_img_path_list = batch_img_path_list[b:b+args.gpu_batch_size]
            b_results = segment_batched_image(b_img_path_list, model, class_names)
            results.extend(b_results)
                
        input_i = 0
        for result_i in range(len(results)):
            separated_results = separate_results(batch_img_list[result_i], results[result_i], class_names)
            person_count, person_masks, category_labels, category_name_labels = separate_person_others( \
                separated_results, person_thresh=args.person_thresh, other_thresh=args.other_thresh)
            while batch_row_list[input_i]["valid_image"] == False:
                input_i += 1
            batch_row_list[input_i]["person_count"] = person_count
            batch_row_list[input_i]["person_masks"] = person_masks
            batch_row_list[input_i]["category_labels"] = category_labels
            batch_row_list[input_i]["category_name_labels"] = category_name_labels
            input_i += 1
                
        # save results for each row
        input_i = 0
        for j in range(idx, batch_end_index):
            instance_save_path = os.path.join(args.output_path, "solo_"+image_names[j].replace(".jpg",".pkl"))
            joblib.dump(batch_row_list[input_i], instance_save_path) # save the row dict
            input_i += 1
            
    print("All done! Saved at path: %s"%(args.output_path))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing of input images \
                                     using SOLO segmentation, saves processed features to pkl files.')
    # SOLO model (default SOLOv2)
    parser.add_argument('--config_file', help='Config file for SOLO model',
                        default='configs/solov2/solov2_r101_dcn_fpn_8gpu_3x.py')
    parser.add_argument('--checkpoint_file', help='Checkpoint file for pretrained SOLO model',
                        default='checkpoints/SOLOv2_R101_DCN_3x.pth')
    # other models: download the checkpoint from model zoo and put it in `checkpoints/`
    # config_file = 'configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py'
    # checkpoint_file = 'checkpoints/DECOUPLED_SOLO_R50_3x.pth'
    # config_file = 'configs/solo/decoupled_solo_r101_fpn_8gpu_3x.py'
    # checkpoint_file = 'checkpoints/DECOUPLED_SOLO_R101_3x.pth'
    
    parser.add_argument('--person_thresh', help='Level of confidence threshold for confirming person category.', type=float, default=0.25)
    parser.add_argument('--other_thresh', help='Level of confidence threshold for confirming other object.', type=float, default=0.25)
    parser.add_argument('--input_path', help='Path to input images folder.',
                        default="../datasets/get_ad_images_cr/creative_ranking/dataset_images")
    parser.add_argument('--output_path', help='Path to save preprocessed output pkl files',                       
            default="../datasets/get_ad_images_cr/preprocessed/solo/")
    
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images to prepare at a time")
    parser.add_argument("--gpu_batch_size", type=int, default=256, help="The GPU inference batchsize")
    
    parser.add_argument("--start_index", type=int, default=0, help="Start Index of the batched processing")
    parser.add_argument("--end_index", type=int, default=0, help="End Index of the batched processing")
    
    args = parser.parse_args()
    
    main(args)
    