import argparse 
import random
import ast
import os
import sys
import copy

import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr_torch.inputs import DenseFeat, SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import *

def parse_args():
    # START Process Arguments:
    parser = argparse.ArgumentParser(description='Train CTR Predictor')
    
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

    parser.add_argument("--exp_name", type=str,
                        default="ctr_predictor_default",
                        help="The experiment name for saving logs and weights.")
    parser.add_argument("--log_path", type=str,
                        default="./logs",
                        help="The logs folder to the trained model experiment")
    parser.add_argument("--data_path", type=str, 
                        default="../datasets/get_ad_images_cr/preprocessed/train_df_with_no_face.pkl",
                        help="The path to trainng data pkl file.")
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

    args = parser.parse_args()
    args.style_vector_dim = ast.literal_eval(args.style_vector_dim)
    args.sparse_features = ast.literal_eval(args.sparse_features)
    args.dense_array_features = ast.literal_eval(args.dense_array_features)
    args.dense_features = ast.literal_eval(args.dense_features)
    # END Process Arguments
    return args  

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
    data["face_latents_raw"] = pd.Series(copy.deepcopy(data["face_latents"].to_dict()))
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
        
    return pred_ans.flatten(), destandardized_pred_ans
    
    
if __name__ == "__main__":
    args = parse_args()
    # Load args with values used by loaded model
    with open(os.path.join(args.log_path, args.exp_name, "args.txt"), 'r') as f:
        loaded_args = json.load(f)
        for key in loaded_args.keys():
            if key in args.__dict__.keys():
                args.__dict__[key] = loaded_args[key]

    # Set random seeds:
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Load Trained model and state_dict for inference
    model = torch.load(os.path.join(args.log_path, args.exp_name, "ctr_predictor_model.h5"))
    model.load_state_dict(torch.load(os.path.join(args.log_path, args.exp_name, "ctr_predictor_best_ckpt.h5")))
    
    # Load original train data and prepare features
    data = joblib.load(args.data_path)
    data = data.reset_index()
    feature_names, label_encoder_dict = prepare_features(args, data.copy(deep=True))
    
    # TODO: replace data with actual new generated data
    # new_data = joblib.load(args.new_data_path) 
    new_data = data.copy(deep=True)
    train_data, test_data = train_test_split(new_data, test_size=args.test_split, random_state=args.random_seed)
    # Filter data with number of faces
    if not args.not_filter_zero_face:
        new_data = new_data[new_data["face_count"] > 0]
        train_data = train_data[train_data["face_count"] > 0]
        test_data = test_data[test_data["face_count"] > 0]
    
    # Predict on New Data
    pred_ans, destandardized_pred_ans = inference_batched(args, 
                        test_data.sample(n=10), model, feature_names, label_encoder_dict)
    print(destandardized_pred_ans)
    print(test_data["ctr"])
    