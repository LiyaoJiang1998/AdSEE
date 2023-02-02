'''
Dataframe Columns: (cr dataset)
    ["image_name", "product_name", 
     "ctr", "log_ctr", "s_log_ctr", 
     "category_labels", "category_name_labels",
     "face_count", "face_latents", 
     "img_embedding"]
'''
import argparse 
import random
import ast
import os
import sys

import json
import joblib
import scipy.stats
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score, max_error, ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr_torch.inputs import DenseFeat, SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import *
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint

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
    
    # Model, Target, and Feature Selection Args
    parser.add_argument("--ctr_model", type=str, default='AutoInt', 
                        choices=["DeepFM", "CCPM", "PNN", "WDL", "MLR", "NFM", 
                                "AFM", "DCNMix", "xDeepFM", "AutoInt", "ONN", 
                                 "FiBiNET", "IFM", "DIFM", "AFN"],
                        help="Select the CTR Predictor Model to use")
    parser.add_argument("--target", type=str, default='s_log_ctr', 
                        choices=["ctr", "log_ctr", "s_log_ctr"],
                        help="Select the target/response (CTR) type") # Note: group_n_log_ctr is used for v4 dataset
    parser.add_argument("--sparse_features", type=str, default='["face_count", "product_name"]',
                        help="The sparse_features list")
    parser.add_argument("--dense_array_features", type=str, 
                        default='["face_latents", "img_embedding"]',
                        help="The dense_array_features list")
    parser.add_argument("--dense_features", type=str, default='[]',
                        help="The dense_features list")
    parser.add_argument("--sigmoid_output", action='store_true',
                        help="Use this flag to use sigmoid on output, so output will be limited to 0 and 1")
    
    parser.add_argument("--random_seed", type=int, default=1024, help="The random seed for reproducability.")
    parser.add_argument("--num_epoch", type=int, default=18, help="The number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="The batch size.")
    parser.add_argument("--test_split", type=float, default=0.2, help="The portion of test data")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="The portion of val data within  the train split")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="The learning rate of optimizer.")
    
    parser.add_argument("--exp_name", type=str,
                        default="ctr_predictor_default",
                        help="The experiment name for saving logs and weights.")
    parser.add_argument("--log_path", type=str,
                        default="./logs",
                        help="The logs folder to save the experiments")
    parser.add_argument("--data_path", type=str, 
                        default="../datasets/get_ad_images_cr/preprocessed/train_df_with_no_face.pkl",
                        help="The path to data pkl file.")
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
    assert type(args.style_vector_dim) == tuple and len(args.style_vector_dim) == 2
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
    
def load_data(args):
    data = joblib.load(args.data_path)
    data = data.reset_index()
    
    data["face_latents"] = convert_face_latent(args, data["face_latents"])
    return data

def prepare_features(args, data):   
    # 1. Label Encoding for sparse features
    for feat in args.sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    
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
    
    # 3.generate input data for model
    train, test = train_test_split(data, test_size=args.test_split, random_state=args.random_seed)
    
    # Filter train and test data with number of faces
    if not args.not_filter_zero_face:
        train = train[train["face_count"] > 0]
        test = test[test["face_count"] > 0]

    train_model_input = {name: np.array(train[name].tolist()) if (name in args.dense_array_features) \
                         else train[name] for name in feature_names}
    test_model_input = {name: np.array(test[name].tolist()) if (name in args.dense_array_features) \
                        else test[name] for name in feature_names}
    
    # process the segmentation "category_labels" sequence feature
    train_category_labels_list = train["category_labels"].values.tolist()
    train_category_labels_list = pad_sequences(train_category_labels_list,
                                               maxlen=args.category_labels_max_len, padding='post')
    test_category_labels_list = test["category_labels"].values.tolist()
    test_category_labels_list = pad_sequences(test_category_labels_list, 
                                              maxlen=args.category_labels_max_len, padding='post')
    train_model_input["category_labels"] = train_category_labels_list
    test_model_input["category_labels"] = test_category_labels_list
    
    return data, train, test, train_model_input, test_model_input, \
            linear_feature_columns, dnn_feature_columns, feature_names
    
def prepare_model(args, linear_feature_columns, dnn_feature_columns, feature_names):
    # 4.Define Model,compile
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    
    # Create Model
    ctr_model = eval(args.ctr_model)
    if args.sigmoid_output:
        task = "binary"
    else:
        task = "regression"
        
    if args.ctr_model in ["WDL", "DeepFM", "MLR", "NFM", "DCN", "DCNMix", 
                          "xDeepFM", "AutoInt", "ONN", "FiBiNET", "IFM", "DIFM", "AFN"]:
        model = ctr_model(linear_feature_columns, dnn_feature_columns, task=task, seed=args.random_seed,
                       device=device, gpus=list(range(torch.cuda.device_count())))
    elif args.ctr_model in ["CCPM", "AFM"]:
        dnn_feature_columns = [x for x in dnn_feature_columns if type(x)!=DenseFeat]
        model = ctr_model(linear_feature_columns, dnn_feature_columns, task=task, seed=args.random_seed,
                       device=device, gpus=list(range(torch.cuda.device_count())))
    elif args.ctr_model in ["PNN"]:
        model = ctr_model(dnn_feature_columns, task=task, seed=args.random_seed,
                       device=device, gpus=list(range(torch.cuda.device_count())))
    elif args.ctr_model in ["DIN", "DIEN"]:
        # Do Not Run, Error: ZeroDivisionError: float division by zero
        model = ctr_model(dnn_feature_columns, [], task=task, seed=args.random_seed,
                       att_weight_normalization=True, 
                       device=device, gpus=list(range(torch.cuda.device_count())))
    else:
        raise ValueError("Model creation is not implemented.")
        
    print(args)
    print(feature_names)
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss = torch.nn.functional.mse_loss
    model.compile(optimizer, loss, metrics=["mse"])
    return model
    
    
if __name__ == "__main__":
    args = parse_args()
    
    # Set Logger
    if not os.path.exists(os.path.join(args.log_path, args.exp_name)):
        os.makedirs(os.path.join(args.log_path, args.exp_name))
    sys.stdout = Logger(os.path.join(args.log_path, args.exp_name, "log.txt"))    
    # Save args used to file
    with open(os.path.join(args.log_path, args.exp_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Set random seeds:
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Load and process dataset pkl file
    data = load_data(args)
    log_ctr_mean = np.log(data["ctr"]).mean()
    log_ctr_std = np.log(data["ctr"]).std()
    target = [args.target]
    
    # Process features
    data, train, test, \
            train_model_input, test_model_input, \
            linear_feature_columns, dnn_feature_columns, \
            feature_names = prepare_features(args, data)
    
    # Compile CTR Predictor Model
    model = prepare_model(args, linear_feature_columns, dnn_feature_columns, feature_names)
    
    # Train
    es = EarlyStopping(monitor='val_mse', min_delta=0, verbose=1, patience=30, mode='min')
    mdckpt = ModelCheckpoint(filepath=os.path.join(args.log_path, args.exp_name, "ctr_predictor_best_ckpt.h5"), \
                             monitor='val_mse', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    history = model.fit(train_model_input, train[target].values,
                        batch_size=args.batch_size, epochs=args.num_epoch,
                        verbose=2, validation_split=args.val_split,
                        callbacks=[es,mdckpt])
    
    # Save End model
    torch.save(model.state_dict(), os.path.join(args.log_path, args.exp_name, "ctr_predictor_end_ckpt.h5"))
    torch.save(model, os.path.join(args.log_path, args.exp_name, "ctr_predictor_model.h5"))
    
    # Load Best model before evluation
    model.load_state_dict(torch.load(os.path.join(args.log_path, args.exp_name, "ctr_predictor_best_ckpt.h5")))
    
    # Predict on Train Data
    pred_ans = model.predict(train_model_input, batch_size=args.batch_size)
    if target[0] == "ctr":
        destandardized_pred_ans = pred_ans.flatten()
    elif target[0] == "log_ctr":
        destandardized_pred_ans = np.exp(pred_ans.squeeze())
    elif target[0] == "s_log_ctr":
        destandardized_pred_ans = np.exp(((pred_ans.squeeze() * log_ctr_std) + log_ctr_mean))
    
    # Evaluate Train Result
    x, y = train["ctr"].values, destandardized_pred_ans
    print("destandardized CTR train variance_explained", round(explained_variance_score(x, y), 7))
    print("destandardized CTR train max error", round(max_error(x, y),7))
    print("destandardized CTR train RMSE", round(mean_squared_error(x, y, squared=False), 7))
    print("destandardized CTR train MSE", round(mean_squared_error(x, y), 7))
    print("destandardized CTR train MAE", round(mean_absolute_error(x, y), 7))
    print("destandardized CTR train MAPE", round(mean_absolute_percentage_error(x, y), 7))
    print("destandardized CTR train NDCG@10", round(ndcg_score([x],[y], k=10), 7))
    print("destandardized CTR train NDCG@50", round(ndcg_score([x],[y], k=50), 7))
    print(round(explained_variance_score(x, y), 7))
    print(round(max_error(x, y),7))
    print(round(mean_squared_error(x, y, squared=False), 7))
    print(round(mean_squared_error(x, y), 7))
    print(round(mean_absolute_error(x, y),7))
    print(round(mean_absolute_percentage_error(x, y), 7))
    print(round(ndcg_score([x],[y], k=10), 7))
    print(round(ndcg_score([x],[y], k=50), 7))
    
    # Predict on Test Data
    pred_ans = model.predict(test_model_input, batch_size=args.batch_size)
    if target[0] == "ctr":
        destandardized_pred_ans = pred_ans.flatten()
    elif target[0] == "log_ctr":
        destandardized_pred_ans = np.exp(pred_ans.squeeze())
    elif target[0] == "s_log_ctr":
        destandardized_pred_ans = np.exp(((pred_ans.squeeze() * log_ctr_std) + log_ctr_mean))
    
    # Evaluate Test Result
    x, y = test["ctr"].values, destandardized_pred_ans
    print("destandardized CTR test variance_explained", round(explained_variance_score(x, y), 7))
    print("destandardized CTR test max error", round(max_error(x, y), 7))
    print("destandardized CTR test RMSE", round(mean_squared_error(x, y, squared=False), 7))
    print("destandardized CTR test MSE", round(mean_squared_error(x, y), 7))
    print("destandardized CTR test MAE", round(mean_absolute_error(x, y),7))
    print("destandardized CTR test MAPE", round(mean_absolute_percentage_error(x, y), 7))
    print("destandardized CTR test NDCG@10", round(ndcg_score([x],[y], k=10), 7))
    print("destandardized CTR test NDCG@50", round(ndcg_score([x],[y], k=50), 7))

    # Correlation on Test Set
    corr_r = scipy.stats.pearsonr(x, y)[0] # Pearson's r
    corr_rho = scipy.stats.spearmanr(x, y)[0] # Spearman's rho
    corr_tau = scipy.stats.kendalltau(x, y)[0] # Kendall's tau
    print("Pearson's R correlation:", round(corr_r, 7))
    print("Spearman's rho correlation:", round(corr_rho, 7))
    print("Kendall's tau correlation:", round(corr_tau, 7))
    
    print(round(explained_variance_score(x, y), 7))
    print(round(max_error(x, y),7))
    print(round(mean_squared_error(x, y, squared=False), 7))
    print(round(mean_squared_error(x, y), 7))
    print(round(mean_absolute_error(x, y),7))
    print(round(mean_absolute_percentage_error(x, y), 7))
    print(round(ndcg_score([x],[y], k=10), 7))
    print(round(ndcg_score([x],[y], k=50), 7))
    
    print(round(corr_r, 7))
    print(round(corr_rho, 7))
    print(round(corr_tau, 7))
