'''
For each text in "all_ad_images.csv", get the embedding from bert

Output format: each row dict is saved to a pkl file.
row = { "idx": idx,
        "valid_text": True
        "image_name": image_name,
        "name_embedding": name_embedding}
        
bert-as-service from: https://github.com/hanxiao/bert-as-service
start server first, run:
    cd server/
    sh starup.sh
'''
import argparse
from argparse import Namespace
import joblib
from tqdm import tqdm
# from pandarallel import pandarallel

import os
import numpy as np
import pandas as pd
from bert_serving.client import BertClient

def save_row_bert_embedding(args, row, name_embedding, description_embedding):
    # Check if row is already finished
    instance_save_path = os.path.join(args.output_path, 
                        "bert_"+row['image_name'].replace(".jpg",".pkl"))
    if os.path.isfile(instance_save_path):
        return

    if "description" not in row:
        row_dict = {
            "idx": row.name,
            "valid_text": False,
            "image_name": row["image_name"],
            "name_text": row["name"],
            "name_embedding": None,
        }

        if name_embedding is not None:
            # case: valid text
            row_dict["valid_text"] = True
            row_dict["name_embedding"] = name_embedding

    elif "description" in row:
        row_dict = {
            "idx": row.name,
            "valid_text": False,
            "image_name": row["image_name"],
            "name_text": row["name"],
            "description_text" : row["description"],
            "name_embedding": None,
            "description_embedding": None, 
        }

        if (name_embedding is not None) and (description_embedding is not None):
            # case: valid text
            row_dict["valid_text"] = True
            row_dict["name_embedding"] = name_embedding
            row_dict["description_embedding"] = description_embedding


    # save each row_dict to individual pkl files
    joblib.dump(row_dict, instance_save_path)

def get_multiple_embedding(args):    
#     pandarallel.initialize() # use pandarallel to do multiprocessing
    bc = BertClient()
    
    # Create Ouput Folder if not exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    # run on text from input .csv file, and save each embeddings to pkl file
    input_df = pd.read_csv(args.input_path, sep='\t', header=0, index_col=0)
    name_texts = list(input_df.name)
    name_embeddings = bc.encode(name_texts)
    
    if "description" in input_df.columns:
        description_texts = list(input_df.description)
        description_embeddings = bc.encode(description_texts)
    
    # Inference is done by the BertClient, now save results to files
    if "description" in input_df.columns:
        input_df.apply(lambda row: save_row_bert_embedding(
                    args, row, name_embeddings[row.name], description_embeddings[row.name]), axis=1)
    else:
        input_df.apply(lambda row: save_row_bert_embedding(
                    args, row, name_embeddings[row.name], None), axis=1)
    
    print("All done! Saved at path: %s"%(args.output_path))
            

def main(args):
    get_multiple_embedding(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--input_path", type=str, \
                        default="../datasets/get_ad_images_cr/all_ad_images.csv",
                        help="The directory to the .csv file contain the text for each ad")
    parser.add_argument("--output_path", type=str,\
                        default="../datasets/get_ad_images_cr/preprocessed/text_embedding/",
                        help="The directory to save the text embeddings.")    
    args = parser.parse_args()
    main(args)
