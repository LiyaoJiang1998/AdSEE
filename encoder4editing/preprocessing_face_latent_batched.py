'''
For each solo_xxx.pkl solo preprocessed person images, get the latent from e4e encoder for each face
Input format:
row_dict = { "idx": idx,
        "valid_image": True
        "image_name": image_name,
        "person_count":person_count,
        "person_masks":person_count,
        "category_labels":person_count,
        "category_name_labels":person_count,}
'''
import argparse
from argparse import Namespace
import joblib
from tqdm import tqdm
import time
import os
import sys
import math
from pandarallel import pandarallel

import numpy as np
import pandas as pd
import scipy
import scipy.ndimage
import PIL
from PIL import Image, ImageFilter
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import dlib

sys.path.append(".")
from utils.alignment_img import align_face_img
from utils.common import tensor2im
from models.psp import pSp  # we use the pSp framework to load the e4e encoder.

def face_img_transform():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    return transform


def load_e4e_model(model_path):
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    # update the training options
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts)
    net.eval()
    
#     net.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net= torch.nn.DataParallel(net)
    net.to(device)
    net = net.module
    return net


def run_alignment(img):    
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face_img(img=img, predictor=predictor) 
#     print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def run_on_batch(inputs, net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, latents = net(inputs.to(device).float(), randomize_noise=False, return_latents=True)
#     if experiment_type == 'cars_encode':
#         images = images[:, :, 32:224, :]
    return images, latents

def get_single_latent(net, img_transform, resize_dims, img):
    try:
        img = run_alignment(img)
    except:
        return None, None
    img.resize(resize_dims)
    transformed_image = img_transform(img)
    
    with torch.no_grad():
        result_images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
        result_image = tensor2im(result_images[0])
        latent = latents[0].cpu().detach().numpy()
    return result_image, latent

def get_batched_latent(net, img_transform, resize_dims, img_list):
    transformed_image_list = []
    invalid_index_list = []
    for index, img in enumerate(img_list):
        try:
            img = run_alignment(img)
        except:
            invalid_index_list.append(index)
            continue

        img.resize(resize_dims)
        transformed_image = img_transform(img)
        transformed_image_list.append(transformed_image)
    
    if transformed_image_list:
        with torch.no_grad():
            result_images, latents = run_on_batch(torch.stack(transformed_image_list, dim=0), net)
        
    result_image_list = []
    latent_list = []
    next_valid_i = 0
    for i in range(len(img_list)):
        if i in invalid_index_list:
            result_image_list.append(None)
            latent_list.append(None)
        else:
            result_image = tensor2im(result_images[next_valid_i])
            latent = latents[next_valid_i].cpu().detach().numpy()
            result_image_list.append(result_image)
            latent_list.append(latent)
            next_valid_i = next_valid_i + 1
            
    return result_image_list, latent_list

def get_separated_img(i, row, args):
    separated_img_list = []
    separated_img_locations_list = []
    if not row["valid_image"]:
        # case: invalid image
        row["face_count"] = None
        row["face_latents"] = None
    else:
        if int(row["person_count"]) <= 0:
            # case: valid image, no person
            row["face_count"] = 0
            row["face_latents_index"] = []
            row["face_latents"] = []

        else:
            # case: valid image, with person
            row["face_count"] = 0
            row["face_latents_index"] = []
            row["face_latents"] = []
            for j in range(int(row["person_count"])):
                # obtain masked human image (separated_img)
                cur_mask_bool = row["person_masks"][j] # retrieve human mask
                input_img_path = os.path.join(args.images_path, row["image_name"])
                input_img = cv2.imread(input_img_path)
                h, w, _ = input_img.shape
                separated_img = np.ones((h,w,3), dtype="uint8") * 119 # blank image background
                separated_img = cv2.GaussianBlur(input_img,(31,31), 0, cv2.BORDER_DEFAULT) # blurred image background
                separated_img[cur_mask_bool] = input_img[cur_mask_bool] # add instance to the background

                separated_img_list.append(separated_img)
                separated_img_locations_list.append((i,j))
    
    row["separated_img_list"] = separated_img_list
    row["separated_img_locations_list"] = separated_img_locations_list
    return row

def get_multiple_latent_batched(net, img_transform, resize_dims, args):
    # run on face images from input pkl files, and save each latents to pkl files
    
    # Create Ouput Folder if not exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    image_names = [f for f in os.listdir(args.images_path) if f.endswith(".jpg")]
    image_names = sorted(image_names, key=lambda x: int(os.path.splitext(x)[0]))
    # Process each image
    if args.end_index <= 0:
        args.end_index = len(image_names)
    for idx in tqdm(range(args.start_index, args.end_index, args.batch_size)):
        batch_end_index = idx+args.batch_size if (idx+args.batch_size) <= len(image_names) else len(image_names)
        
        batch_already_done = True
        for j in range(idx, batch_end_index):
            instance_save_path = os.path.join(args.output_path, "e4e_"+image_names[j].replace(".jpg",".pkl"))
            if not os.path.isfile(instance_save_path):
                batch_already_done = False
        if batch_already_done:
            continue # If this batch all done before, go to next batch
                    
        batch_load_path_list = []
        for j in range(idx, batch_end_index):
            instance_load_path = os.path.join(args.input_path, "solo_"+image_names[j].replace(".jpg",".pkl"))
            batch_load_path_list.append(instance_load_path)
        batch_load_path_df = pd.DataFrame({"instance_load_path": batch_load_path_list})
        batch_row_list = batch_load_path_df.parallel_apply(lambda row: joblib.load(row["instance_load_path"]), axis=1)
        batch_row_list = batch_row_list.to_list()
        
        # Method 1:
        batch_row_list_df = pd.DataFrame(batch_row_list)
        apply_result_df = batch_row_list_df.parallel_apply(lambda row: get_separated_img(row.name, row, args), axis=1)
        
        batch_separated_img_list = apply_result_df["separated_img_list"]
        batch_separated_img_locations_list = apply_result_df["separated_img_locations_list"]
        batch_row_list_df = apply_result_df.drop(['separated_img_list', 'separated_img_locations_list'], axis=1)
        
        batch_row_list = batch_row_list_df.to_dict('records')
        batch_separated_img_list = batch_separated_img_list.apply(
                                        pd.Series).stack().reset_index(drop = True).to_list()
        batch_separated_img_locations_list = batch_separated_img_locations_list.apply(
                                        pd.Series).stack().reset_index(drop = True).to_list()
                              
        # obtain latent code for the batch, split by gpu_batch_size to fit into GPU
        latent_list = []
        for b in range(0, len(batch_separated_img_list), args.gpu_batch_size):
            b_separated_img_list = batch_separated_img_list[b:b+args.gpu_batch_size]
            b_result_image_list, b_latent_list = get_batched_latent(net, img_transform, resize_dims, b_separated_img_list)
            latent_list.extend(b_latent_list)
        
        # Fill in batched results
        for index in range(len(latent_list)):
            latent = latent_list[index]
            i, j = batch_separated_img_locations_list[index]
            if latent is not None:
                # successfully found face, aligned and obtained latent
                batch_row_list[i]["face_count"] += 1
                batch_row_list[i]["face_latents_index"].append(j)
                batch_row_list[i]["face_latents"].append(latent)
        
        # save results for each row
        batch_save_path_list = []
        input_i_list = []
        input_i = 0
        for j in range(idx, batch_end_index):
            instance_save_path = os.path.join(args.output_path, "e4e_"+image_names[j].replace(".jpg",".pkl"))
            batch_save_path_list.append(instance_save_path)
            input_i_list.append(input_i)
            input_i += 1
        batch_save_path_df = pd.DataFrame({"instance_save_path": batch_save_path_list, "input_i": input_i_list})
        batch_save_path_df.parallel_apply(lambda row: 
                            joblib.dump(batch_row_list[row["input_i"]], row["instance_save_path"]), axis=1) # save the row dict
            
    print("All done! Saved at path: %s"%(args.output_path))
    
    
def main(args):
    pandarallel.initialize(nb_workers=24, progress_bar=False)
    
    # define transform for face images
    img_transform = face_img_transform()
    resize_dims = (256, 256)
    
    # load e4e model
    net = load_e4e_model(args.model_path)
    print('Model successfully loaded!')
    
    if 'shape_predictor_68_face_landmarks.dat' not in os.listdir():
        download_cmd = "!wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n" + \
                        "!bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2"
        raise ValueError("Download using the following command: "+download_cmd)
        
    # try on single image:
#     img = cv2.imread(os.path.join(args.images_path, "42.jpg"))
#     result_image, latent = get_single_latent(net, img_transform, resize_dims, img)
    
    # run on face images from input pkl file
    get_multiple_latent_batched(net, img_transform, resize_dims, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--images_path", type=str, default="../datasets/get_ad_images_cr/creative_ranking/dataset_images/", help="path to full images")
    parser.add_argument("--input_path", type=str, \
                        default="../datasets/get_ad_images_cr/preprocessed/solo/",
                        help="The directory to individual solo_xxx.pkl files contain solo preprocessed images data")
    parser.add_argument("--output_path", type=str,\
                        default="../datasets/get_ad_images_cr/preprocessed/e4e/",
                        help="The directory to save the face latent codes as individual .pkl files")
    parser.add_argument("--model_path", type=str, default="pretrained_models/e4e_ffhq_encode.pt", help="path to e4e encoder model checkpoint")
        
    parser.add_argument("--batch_size", type=int, default=128, help="The number of images to prepare at a time")
    parser.add_argument("--gpu_batch_size", type=int, default=32, help="The GPU inference batchsize")
    
    parser.add_argument("--start_index", type=int, default=0, help="Start Index of the batched processing")
    parser.add_argument("--end_index", type=int, default=0, help="End Index of the batched processing")
    
    args = parser.parse_args()
    main(args)
