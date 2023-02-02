import argparse
from argparse import Namespace
import random
import ast
import os
import sys
import copy
from tqdm import tqdm
import torch.multiprocessing as mp

import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from scipy.special import expit

from deepctr_torch.inputs import DenseFeat, SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import *
import pygad

# e4e encoder (using psp framework) and latent editors
sys.path.append('../encoder4editing')
import PIL.Image
import dlib
import cv2
from models.psp import pSp
from editings.sefa import factorize_weight
from editing_utils import latents_to_images, solution_to_faces_images
from editing_utils import inverse_align_face, merge_persons_with_mask, images_side_by_side

class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass  
    
def parse_args():
    # START Process Arguments:
    parser = argparse.ArgumentParser(description='Train CTR Predictor')
    parser.add_argument("--num_processes", type=int, default=1, 
                        help="Number of processes for multiprocessing. Default 1")
    parser.add_argument("--save_interval", type=int, default=100, 
                        help="Interval to save the fitness_df")
    parser.add_argument("--start_index", type=int, default=0, 
                        help="Start index of the selected data rows range (included)")
    parser.add_argument("--end_index", type=int, default=0, 
                        help="End index of the selected data rows range (excluded)")
    parser.add_argument("--edit_samples", type=int, default=0, 
                        help="How many images in the test set to edit, if 0 then all will be used")
    
    # Model, Target, and Feature Selection Args
    parser.add_argument("--ctr_model", type=str, default='AutoInt', 
                        choices=["DeepFM", "CCPM", "PNN", "WDL", "MLR", "NFM", 
                                "AFM", "DCNMix", "xDeepFM", "AutoInt", "ONN", 
                                 "FiBiNET", "IFM", "DIFM", "AFN"],
                        help="Select the CTR Predictor Model to use")
    parser.add_argument("--target", type=str, default='s_log_ctr', 
                        choices=["ctr", "log_ctr", "s_log_ctr"],
                        help="Select the target/response (CTR) type")
    parser.add_argument("--sparse_features", type=str, default='["face_count", "product_name"]',
                        help="The sparse_features list")
    parser.add_argument("--dense_array_features", type=str, 
                        default='["face_latents", "img_embedding"]',
                        help="The dense_array_features list")
    parser.add_argument("--dense_features", type=str, default='[]',
                        help="The dense_features list")
    
    parser.add_argument("--random_seed", type=int, default=1024, help="The random seed for reproducability.")
    parser.add_argument("--batch_size", type=int, default=256, help="The batch size.")
    parser.add_argument("--test_split", type=float, default=0.2, help="The portion of test data")
    parser.add_argument("--log_ctr_mean", type=float, default=-3.6582919758327033)
    parser.add_argument("--log_ctr_std", type=float, default=0.6593969081338986)

    parser.add_argument("--ga_exp_name", type=str,
                        default="ga_default",
                        help="The experiment name for saving logs and weights.")
    parser.add_argument("--ga_log_path", type=str,
                        default="./results",
                        help="The results folder to the GA optimization experiment")
    parser.add_argument("--exp_name", type=str,
                        default="ctr_predictor_default",
                        help="The experiment name for saving logs and weights.")
    parser.add_argument("--log_path", type=str,
                        default="./logs",
                        help="The logs folder to save the experiments")
    parser.add_argument("--data_path", type=str, 
                        default="../datasets/get_ad_images_cr/preprocessed/train_df_with_no_face.pkl",
                        help="The path to trainng data pkl file.")
    parser.add_argument("--org_images_path", type=str, 
                        default="../datasets/get_ad_images_cr/creative_ranking/dataset_images",
                        help="The path to the original image files.")
    parser.add_argument("--person_masks_path", type=str, 
                        default="../datasets/get_ad_images_cr/preprocessed/e4e/",
                        help="The path to the original image files.")
    parser.add_argument('--not_filter_zero_face', action='store_true',
                      help="Use this flag, if want to include images even with zero faces in them.")  
    parser.add_argument("--category_labels_max_len", type=int, default=80,
                        help="The number of instance classification classes (number of classes in COCO)")
    parser.add_argument("--max_num_person", type=int, default=5, help="The max number of face latents.")
    parser.add_argument("--style_vector_dim", type=str, default="(18, 512)",
                        help="The dimension tuple string, e.g. (18, 512)")
    parser.add_argument("--style_vector_method", type=str, default='max_pooling', 
                        choices=["pad_fix_len", "max_pooling", "average_pooling", "aggregation"],
                        help="Method to process face lentent to make it fixed length?")
    
    # GA algorithom Args:
    parser.add_argument("--init_range_low", type=float, default=-0.1, help="The random initialize value lower limit")
    parser.add_argument("--init_range_high", type=float, default=0.1, help="The random initialize value upper limit")
    parser.add_argument("--gene_range_low", type=float, default=-1.5, help="Overall Edit value lower limit")
    parser.add_argument("--gene_range_high", type=float, default=1.5, help="Overall Edit value upper limit")
    parser.add_argument("--gene_range_step", type=float, default=0.01, help="Edit coefficient step value")
    parser.add_argument("--random_mutation_min_val", type=float, default=-0.01, help="random mutation change lower limit")
    parser.add_argument("--random_mutation_max_val", type=float, default=0.01, help="random mutation change upper limit")
    parser.add_argument("--sol_per_pop", type=int, default=30, help="GA parameter")
    parser.add_argument("--num_generations", type=int, default=5, help="GA parameter")
    parser.add_argument("--num_parents_mating", type=int, default=10, help="GA parameter")
    parser.add_argument("--keep_parents", type=int, default=-1, help="GA parameter")
    parser.add_argument("--mutation_percent_genes", type=int, default=20, help="GA parameter")
    parser.add_argument("--mutation_type", type=str, default="random", help="GA parameter")
    parser.add_argument("--parent_selection_type", type=str, default="rank", help="GA parameter")
    parser.add_argument("--crossover_type", type=str, default="uniform", help="GA parameter")
    
    # e4e generator and editor Args:
    parser.add_argument("--face_align_model_path", type=str, \
                        default="checkpoints/shape_predictor_68_face_landmarks.dat", \
                        help="The path to dlib face alignment model checkpoint")
    parser.add_argument("--e4e_model_path", type=str, \
                        default="checkpoints/e4e_ffhq_encode.pt", \
                        help="The path to e4e ffhq generator model checkpoint")
    parser.add_argument("--sefa_layers", type=str, default="list(range(3,6))", \
                        help='The layer indicies to allow SeFa editing on (allowed for From 0~16).')
#     parser.add_argument("--sefa_layers", type=str, default="list(range(3,6))", \
#                         choices=["all", "list(range(0,2))", "list(range(2,6))", "list(range(6,14))"], \
#                         help='The layer indicies to allow SeFa editing on (allowed for From 0~16).')
    parser.add_argument("--sefa_top_k", type=int, default=20, \
                        help="The number of top K SeFa edit directions to consider")
    parser.add_argument("--sefa_k_list", type=str, default="None", \
                        help="List of indexs of which directions to use.")
    
    args = parser.parse_args()
    args.style_vector_dim = ast.literal_eval(args.style_vector_dim)
    args.sparse_features = ast.literal_eval(args.sparse_features)
    args.dense_array_features = ast.literal_eval(args.dense_array_features)
    args.dense_features = ast.literal_eval(args.dense_features)
    if args.crossover_type == "None":
        args.crossover_type = None
    if args.mutation_type == "None":
        args.mutation_type = None
    if args.sefa_layers != "all": 
        args.sefa_layers = eval(args.sefa_layers)
    if args.sefa_k_list == "None":
        args.sefa_k_list = None
    else:
        args.sefa_k_list = eval(args.sefa_k_list)
        
    # END Process Arguments
    return args
  
def get_sefa_edit_directions(args, e4e_net):
    '''
    return an np.array of SeFa editing directions (by factorizing FFHQ Face e4e generator)
    '''
    print('"e4e_ffhq_encode" Model successfully loaded!')
    layers, boundaries, values = factorize_weight(e4e_net.decoder, args.sefa_layers)
    layers = [x+1 for x in layers] # exclude first and last layer.
    print("editing allowed on layers:", layers)
    
    if args.sefa_k_list is None: 
        edit_directions = np.zeros((args.sefa_top_k, args.style_vector_dim[0], 
                                    args.style_vector_dim[1]),dtype=np.float32)
        edit_directions[:,layers,:] += np.expand_dims( \
                boundaries[:args.sefa_top_k], axis=1) # broadcast (k,1,512) to (k, len(layers), 512)
    else:
        edit_directions = np.zeros((len(args.sefa_k_list), args.style_vector_dim[0], 
                                    args.style_vector_dim[1]),dtype=np.float32)
        edit_directions[:,layers,:] += np.expand_dims( \
                boundaries[np.array(args.sefa_k_list)], axis=1) # broadcast (k,1,512) to (k, len(layers), 512)
    return edit_directions
    
def convert_face_latent(args, face_latents):
    assert len(args.style_vector_dim) == 2
    # Choices: ["pad_fix_len", "max_pooling", "average_pooling", "aggregation"]
    if args.style_vector_method == "pad_fix_len":
        # Zero-Pad face_latents to 5 person* 18*512..., to make it fixed length
        final_size = args.max_num_person * args.style_vector_dim[0] * args.style_vector_dim[1]
        face_latents = face_latents.apply(lambda x : 
                                        np.array(np.pad(x, (0, final_size - x.size),
                                        mode='constant', constant_values=0),dtype=np.float32))
    elif args.style_vector_method == "max_pooling":
        face_latents = face_latents.apply(lambda x : 
                                        np.reshape(np.array(x, dtype=np.float32), 
                                                   (-1, args.style_vector_dim[0], args.style_vector_dim[1])))
        face_latents = face_latents.apply(lambda x : 
                                        np.amax(np.array(x, dtype=np.float32), axis=0) if len(x)!=0
                                        else np.zeros((args.style_vector_dim[0], args.style_vector_dim[1]),
                                                        dtype=np.float32))
    elif args.style_vector_method == "average_pooling":
        face_latents = face_latents.apply(lambda x : 
                                        np.reshape(np.array(x, dtype=np.float32), 
                                                   (-1, args.style_vector_dim[0], args.style_vector_dim[1])))
        face_latents = face_latents.apply(lambda x : 
                                        np.mean(np.array(x, dtype=np.float32), axis=0) if len(x)!=0
                                        else np.zeros((args.style_vector_dim[0], args.style_vector_dim[1]),
                                                        dtype=np.float32))
    elif args.style_vector_method == "aggregation":
        face_latents = face_latents.apply(lambda x : 
                                        np.reshape(np.array(x, dtype=np.float32), 
                                                   (-1, args.style_vector_dim[0], args.style_vector_dim[1])))
        face_latents = face_latents.apply(lambda x : 
                                        np.sum(np.array(x, dtype=np.float32), axis=0) if len(x)!=0
                                        else np.zeros((args.style_vector_dim[0], args.style_vector_dim[1]),
                                                        dtype=np.float32))
    else:
        raise ValueError("Invalid style vector processing option:", args.style_vector_method)
    
    face_latents = face_latents.apply(lambda x : x.flatten())
    return face_latents
    
def prepare_features(args, data):
    '''
    Inference Need the folloiwing columns: 
    ['product_name', 'category_labels', 'face_count', 'face_latents', 'img_embedding']
    '''
    data["face_latents_raw"] = pd.Series(copy.deepcopy(data["face_latents"].to_dict()))
    data["face_latents"] = convert_face_latent(args, data["face_latents"])

    # 1. Label Encoding for sparse features
    label_encoder_dict = {}
    for feat in args.sparse_features:
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(data[feat])
        label_encoder_dict[feat] = label_encoder
        data[feat] = label_encoder.transform(data[feat])
    
    # 2. count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in args.sparse_features] + \
                                [DenseFeat(feat, len(data[feat][0])) for feat in args.dense_array_features] + \
                                [DenseFeat(feat, 1,) for feat in args.dense_features]
    
    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('category_labels',
                                    vocabulary_size= args.category_labels_max_len),
                                    maxlen=args.category_labels_max_len, combiner='mean')]
    
    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)    
    return feature_names, label_encoder_dict

def inference_batched(args, data, model, feature_names, label_encoder_dict):
    # 1. Preprocess face latents style vector, for the new inference data
    data["face_latents"] = convert_face_latent(args, data["face_latents"])
    
    # 2. Label Encoding for sparse features
    for feat in args.sparse_features:
        data[feat] = label_encoder_dict[feat].transform(data[feat])
        
    # 3. generate input data for model
    model_input = {name: np.array(data[name].tolist()) if (name in args.dense_array_features) \
                        else data[name] for name in feature_names}
        
    # 4. Process the segmentation "category_labels" sequence feature
    category_labels_list = data["category_labels"].values.tolist()
    category_labels_list = pad_sequences(category_labels_list, 
                                              maxlen=args.category_labels_max_len, padding='post')
    model_input["category_labels"] = category_labels_list
    
    # 5. Predict on batched new data
    pred_ans = model.predict(model_input, batch_size=args.batch_size)

    target = [args.target]
    if target[0] == "ctr":
        destandardized_pred_ans = pred_ans.flatten()
    elif target[0] == "log_ctr":
        destandardized_pred_ans = np.exp(pred_ans.squeeze())
    elif target[0] == "s_log_ctr":
        destandardized_pred_ans = np.exp(((pred_ans.squeeze() * args.log_ctr_std) + args.log_ctr_mean))
        
    return pred_ans.flatten(), destandardized_pred_ans.flatten()
                            
def optimize_face_latent_GA(args, data, model, feature_names, label_encoder_dict, edit_directions, e4e_net):
    '''
    Optimize the face_latent of one AD image, using Genetic Algorithom
    input:
        args: options for everything
        data: dataframe with length 1, contain the AD to optimize
        model: CTR predictor model
        feature_names: used by CTR predictor model
        label_encoder_dict: used by CTR predictor model
    '''    
    def fitness_function(solution, solution_idx):
        '''
        Use predicted CTR as fitness value.
        Use the edits represented by the solution, modify the latent, run inference and return the predicted CTR 
        solution_face_latent (#face, 18, 512), reshaped_solution (#face, #direction), edit_directions (#direction, 18, 512)
        '''
        solution_data = pd.DataFrame(copy.deepcopy(data.to_dict()))
        reshaped_solution = np.reshape(solution, ((int(solution_data["face_count"][0]), len(edit_directions))))
        # Apply the edits in solution to the face_latents_raw
        solution_face_latent = np.reshape(solution_data['face_latents_raw'][0], \
                        (int(solution_data["face_count"][0]), args.style_vector_dim[0], args.style_vector_dim[1])) 
        for xid in range(solution_face_latent.shape[0]):     # for each face
            for yid in range(reshaped_solution.shape[1]): # for each edit direction
                solution_face_latent[xid, :, :] = solution_face_latent[xid, :, :] \
                                    + reshaped_solution[xid, yid] * edit_directions[yid,:,:]
        solution_data.at[0, 'face_latents'] = solution_face_latent.flatten()
        pred_ans, destandardized_pred_ans = inference_batched(args, solution_data, model, feature_names, label_encoder_dict)

        fitness = destandardized_pred_ans[0]
#         fitness = expit(destandardized_pred_ans[0])
#         fitness = pred_ans[0]
        return fitness
    
    trivial_solution = np.zeros((int(data["face_count"][0]), len(edit_directions))) # unedited image
    original_fitness = fitness_function(trivial_solution, 0) # get predicted CTR for unedited image.
    last_fitness = original_fitness
    def on_generation(ga_instance):
        nonlocal last_fitness
        output_str = "Generation = {generation}".format(generation=ga_instance.generations_completed) + " | "
        output_str += "Fitness    = {fitness}".format(fitness= \
                        ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]) + " | "
        output_str += "Change     = {change}".format(change= \
                        ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness)
        print(output_str)
        last_fitness = ga_instance.best_solution()[1]
    
    print("Start Editing image:", data["index"][0])
    print("Original Fitness:", original_fitness)
    assert len(data) == 1 # data: dataframe with length 1, contain the AD to optimize
    num_genes = len(edit_directions) * int(data["face_count"][0]) # i.e. #directions * #faces
    
    ga_instance = pygad.GA(fitness_func=fitness_function,
                           on_generation=on_generation,
                           stop_criteria="reach_1",
                           num_generations=args.num_generations,
                           sol_per_pop=args.sol_per_pop, 
                           num_genes=num_genes,
                           num_parents_mating=args.num_parents_mating, 
                           keep_parents=args.keep_parents,
                           init_range_low=args.init_range_low,
                           init_range_high=args.init_range_high,
                           gene_space={'low': args.gene_range_low, 
                                       'high': args.gene_range_high, 
                                       'step': args.gene_range_step},
                           random_mutation_min_val=args.random_mutation_min_val,
                           random_mutation_max_val=args.random_mutation_max_val,
                           mutation_percent_genes=args.mutation_percent_genes,
                           mutation_type=args.mutation_type,
                           parent_selection_type=args.parent_selection_type,
                           crossover_type=args.crossover_type)

    # Running the GA to optimize the parameters of the function.
    ga_instance.run()

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    
    # Print out Solution
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Fitness changed by = {fitness_diff}".format(fitness_diff=solution_fitness - original_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    if ga_instance.best_solution_generation != -1:
        print("Best fitness value reached after {best_solution_generation} generations.".format( \
            best_solution_generation=ga_instance.best_solution_generation))
    
    print("Finished Editing image:", data["index"][0])
    
    return solution, solution_fitness, original_fitness

def visualize_face_latent_solution(args, solution, data, edit_directions, e4e_net, face_align_predictor):
    '''
    1st Return: boolean, indicate image load successed or failed
    2nd Return: boolean, indicate any face swap made or no face edit is made
    '''
    inverse_faces_latents = data["face_latents_raw"][0].reshape(
                (int(data["face_count"][0]), args.style_vector_dim[0], args.style_vector_dim[1]))        
    inverse_faces_images = latents_to_images(inverse_faces_latents, e4e_net.decoder)    
    edited_faces_images = solution_to_faces_images(solution, data, edit_directions,
                                                   args.style_vector_dim, e4e_net.decoder)
    try:
        org_image_array = dlib.load_rgb_image(os.path.join(args.org_images_path, data["image_name"][0]))
    except Exception as e:
        print(e)
        return False, False
    org_image = PIL.Image.fromarray(org_image_array.astype('uint8'), 'RGB')
    person_masks_list = joblib.load(os.path.join(
                            args.person_masks_path, "e4e_"+data["image_name"][0].replace(".jpg",".pkl")))
    person_masks_list = person_masks_list["person_masks"]    
    person_masks_list = [x for i,x in enumerate(person_masks_list) 
                         if i in data["face_latents_index"][0]] # keep only persons with detected faces
    original_person_images_list = []
    inverse_person_images_list = []
    edited_person_images_list = []
    for face_i in range(len(person_masks_list)):
        # obtain original person image, with background blurred or blank
        cur_mask_bool = person_masks_list[face_i]
#         person_image = np.ones((org_image_array.shape[0],
#                                 org_image_array.shape[1], 3), dtype="uint8") * 119 # blank image background
        gaussian_kernel_size = (2 * int(0.1*org_image_array.shape[0])) + 1
        person_image = cv2.GaussianBlur(org_image_array,(gaussian_kernel_size, gaussian_kernel_size), 
                                        gaussian_kernel_size, cv2.BORDER_DEFAULT) # blurred image background
        person_image[cur_mask_bool] = org_image_array[cur_mask_bool] # add instance to the background
        original_person_images_list.append(person_image)
        
        # obtain inverse and edited person image, by putting edited face back into person
        # Option 1: inverse_align_face
#         inverse_person_image = inverse_align_face(np.array(inverse_faces_images[face_i]), 
#                            original_person_images_list[face_i], 
#                            face_align_predictor)
#         edited_person_image = inverse_align_face(np.array(edited_faces_images[face_i]), 
#                            original_person_images_list[face_i], 
#                            face_align_predictor)
#         inverse_person_images_list.append(inverse_person_image)
#         edited_person_images_list.append(edited_person_image)
        # Option 2: don't inverse_align_face, directly swap aligned face.
        inverse_person_images_list.append(np.array(inverse_faces_images[face_i]))
        edited_person_images_list.append(np.array(edited_faces_images[face_i]))
        
    whole_inverse_image, _ = merge_persons_with_mask(
                org_image_array, inverse_person_images_list,
                original_person_images_list, person_masks_list, face_align_predictor, erode=3)
    whole_edited_image, any_swap_made = merge_persons_with_mask(
                org_image_array, edited_person_images_list, 
                original_person_images_list, person_masks_list, face_align_predictor, erode=3)
    whole_side_by_side = images_side_by_side([org_image, whole_inverse_image, whole_edited_image], 
                                             org_image.size, concat_axis=0)
    # Save Result Images
    if any_swap_made: # only save results if image/face is actually edited, don't just save trivial/original case.
        path_list = [args.ga_shorthand_path,
                     os.path.join(args.ga_shorthand_path, "original"),
                     os.path.join(args.ga_shorthand_path, "inverse"),
                     os.path.join(args.ga_shorthand_path, "edited"),
                     os.path.join(args.ga_shorthand_path, "compare")]
        for p in path_list:
            if not os.path.exists(p):
                os.makedirs(p)

        org_image.save(os.path.join(args.ga_shorthand_path, 
                                             "original", str(data["index"][0])+"_original.jpg"))
        whole_inverse_image.save(os.path.join(args.ga_shorthand_path, 
                                             "inverse", str(data["index"][0])+"_inverse.jpg"))
        whole_edited_image.save(os.path.join(args.ga_shorthand_path, 
                                             "edited", str(data["index"][0])+"_edited.jpg"))
        whole_side_by_side.save(os.path.join(args.ga_shorthand_path, 
                                             "compare", str(data["index"][0])+"_compare.jpg"))
        print("Visualization is saved for:", data["image_name"][0])
        
    return True, any_swap_made

def job_func(args_in):
    try:
        idx, temp_data, args, model, feature_names, label_encoder_dict, edit_directions, \
                e4e_net, face_align_predictor = args_in
        
        row_dict = {'image_name': temp_data["image_name"][0],
                    'original_fitness': None, 
                    'solution_fitness': None, 
                    'fitness_diff': None,
                    'solution': None,
                    'loading_successed': False,
                    'any_swap_made': False}
    
        if os.path.isfile(os.path.join(args.ga_shorthand_path, "compare", str(temp_data["index"][0])+"_compare.jpg")):
            return row_dict

        solution, solution_fitness, original_fitness = optimize_face_latent_GA(
                    args, temp_data, model, feature_names, label_encoder_dict, edit_directions, e4e_net)

        loading_successed, any_swap_made = visualize_face_latent_solution(
            args, solution, temp_data, edit_directions, e4e_net, face_align_predictor)
        
        row_dict = {'image_name': temp_data["image_name"][0],
                    'original_fitness': original_fitness, 
                    'solution_fitness': solution_fitness, 
                    'fitness_diff': solution_fitness-original_fitness,
                    'solution': solution,
                    'loading_successed': loading_successed,
                    'any_swap_made': any_swap_made}
        
    except Exception as e:
        print(e)
        pass
        
    return row_dict

if __name__ == "__main__":
    args = parse_args()
    vars(args)["ga_shorthand_path"] = os.path.join(args.ga_log_path, args.ga_exp_name)
    
    # Set Logger
    if not os.path.exists(os.path.join(args.ga_shorthand_path)):
        os.makedirs(os.path.join(args.ga_shorthand_path))
    sys.stdout = Logger(os.path.join(args.ga_shorthand_path, "GA_log.txt"))
    
    # Load args with values used by loaded model
    with open(os.path.join(args.log_path, args.exp_name, "args.txt"), 'r') as f:
        loaded_args = json.load(f)
        for key in loaded_args.keys():
            if key in args.__dict__.keys():
                args.__dict__[key] = loaded_args[key]
    
    # Save ga_args used to file
    with open(os.path.join(args.ga_shorthand_path, "ga_args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Set random seeds:
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Load Trained model and state_dict for inference
    model = torch.load(os.path.join(args.log_path, args.exp_name, "ctr_predictor_model.h5"))
    model.load_state_dict(torch.load(os.path.join(
                args.log_path, args.exp_name, "ctr_predictor_best_ckpt.h5")))
    
    # Load original train data and prepare features
    data = joblib.load(args.data_path)
    data = data.reset_index()
    feature_names, label_encoder_dict = prepare_features(args, data.copy(deep=True))
    
    new_data = data.copy(deep=True)
    # TODO: replace data with actual new generated data
    # new_data = joblib.load(args.new_data_path)
    new_data["face_latents_raw"] = pd.Series(copy.deepcopy(new_data["face_latents"].to_dict()))
    train_data, test_data = train_test_split(new_data, test_size=args.test_split, random_state=args.random_seed)
    # Filter out data with zero faces
    if not args.not_filter_zero_face:
        new_data = new_data[new_data["face_count"] > 0]
        train_data = train_data[train_data["face_count"] > 0]
        test_data = test_data[test_data["face_count"] > 0]
    print("Filtered Full/Train/Test dataset sizes:", len(new_data), len(train_data), len(test_data))
    
    # Load face alignment model using dlib
    face_align_predictor = dlib.shape_predictor(args.face_align_model_path)
    
    # Load e4e generator model
    ckpt = torch.load(args.e4e_model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = args.e4e_model_path
    opts= Namespace(**opts)
    e4e_net = pSp(opts)
    e4e_net.eval()
    e4e_net.cuda()
        
    # Edit with SeFa directions
    edit_directions = get_sefa_edit_directions(args, e4e_net)
    

    if args.edit_samples > 0:
        selected_data = test_data.sample(n=args.edit_samples).reset_index(drop=True)
        
    if args.start_index==0 and args.end_index==0:
        # selected_data = train_data.reset_index(drop=True)
        selected_data = selected_data.reset_index(drop=True)
    else:
        # selected_data = train_data.reset_index(drop=True).iloc[args.start_index:args.end_index].reset_index(drop=True)
        selected_data = selected_data.reset_index(drop=True).iloc[args.start_index:args.end_index].reset_index(drop=True)


    
    # seleted_image_names = ["xxx.jpg", "xxx.jpg"]
    # selected_data = test_data.loc[test_data["image_name"].isin(seleted_image_names)].reset_index(drop=True)
    # print(selected_data["image_name"].values)
    
    row_list = []
    # Single Process
    if args.num_processes == 1:    
        for i in tqdm(range(len(selected_data))):
            temp_data = selected_data.iloc[[i]].reset_index(drop=True)
            try:
                solution, solution_fitness, original_fitness = optimize_face_latent_GA(
                            args, temp_data, model, feature_names, label_encoder_dict, edit_directions, e4e_net)        
                loading_successed, any_swap_made = visualize_face_latent_solution(
                    args, solution, temp_data, edit_directions, e4e_net, face_align_predictor)

                row_dict = {'image_name': temp_data["image_name"][0],
                            'original_fitness': original_fitness, 
                            'solution_fitness': solution_fitness, 
                            'fitness_diff': solution_fitness-original_fitness,
                            'solution': solution,
                            'loading_successed': loading_successed,
                            'any_swap_made': any_swap_made}
            except Exception as e:
                row_dict = {'image_name': temp_data["image_name"][0],
                            'original_fitness': None, 
                            'solution_fitness': None, 
                            'fitness_diff': None,
                            'solution': None,
                            'loading_successed': False,
                            'any_swap_made': False}
            row_list.append(row_dict)
            
            if (i+1) % args.save_interval == 0:
                fitness_df = pd.DataFrame(row_list)
                fitness_df.to_csv(os.path.join(args.ga_shorthand_path, "fitness_df.csv"), sep="\t")    
    # Multiple Processes
    else:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        with mp.Pool(processes=args.num_processes) as pool:
            row_list = pool.map(job_func, [(idx, pd.DataFrame([row]).reset_index(drop=True),
                               args, model, feature_names, label_encoder_dict, 
                               edit_directions, e4e_net, face_align_predictor)
                                    for idx, row in selected_data.iterrows()])
            
    fitness_df = pd.DataFrame(row_list)
    fitness_df.to_csv(os.path.join(args.ga_shorthand_path, "fitness_df.csv"), sep="\t")
    print(fitness_df["fitness_diff"].describe())
    print("Fitness Increase Standard Deviation:", fitness_df["fitness_diff"].std())
    print("Average Fitness Increase:", fitness_df["fitness_diff"].mean())
    
    