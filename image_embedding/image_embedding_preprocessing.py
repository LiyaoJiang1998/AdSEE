import argparse
import pandas as pd
import numpy as np
import joblib
import requests
import os
from pandarallel import pandarallel

from PIL import Image
from img2vec_pytorch import Img2Vec


def img_to_embedding(img2vec, img_dir, image_name):
    '''
    Using the Image to Embedding library img2vec_pytorch from https://github.com/christiansafka/img2vec
    install it with "pip install img2vec_pytorch"
    '''
    try:
        code = 1 # default to failed.
        input_img_path = os.path.join(img_dir, image_name)
        input_img = Image.open(input_img_path)  
        embedding = img2vec.get_vec(input_img, tensor=False)
        embedding = embedding.tolist()
        code = 0
        return code, embedding
    except Exception as e:
        print(e)
        print(image_name)
        return 1, None
    
    
def get_row_image_embedding(args, row):
    # share img2vec library resnet-18 image embedding model:
    img2vec = Img2Vec(cuda=True, model='resnet-18', layer='default', layer_output_size=512)
    
    # Check if row is already finished
    instance_save_path = os.path.join(args.output_path, 
                        "image_embedding_"+row['image_name'].replace(".jpg",".pkl"))
    if os.path.isfile(instance_save_path):
        return

    # get 'img_embedding'
    try:
        code, embedding = img_to_embedding(img2vec, args.img_dir, row['image_name']) # call API to get embedding for image
        retry = 0
        while code != 0 and retry <= args.num_retry:
            code, embedding = img_to_embedding(img2vec, args.img_dir, row['image_name'])
            retry += 1
        if code == 0:
            img_embedding = embedding
        else:
            img_embedding = None
    except Exception as e:
        print(e)
        print(row)
        img_embedding = None

    # add to df row
    row_dict = {"idx": row.name,
                "image_name": row['image_name'],
                "img_embedding":img_embedding}

    # save each row_dict to individual pkl files
    joblib.dump(row_dict, instance_save_path)
    return

def main(args):    
    # pandarallel.initialize(progress_bar=True) # use pandarallel to do multiprocessing
    pandarallel.initialize(nb_workers=args.nb_workers, progress_bar=True)
        
    df = pd.read_csv(args.input_path, sep='\t', header=0, index_col="id")
    df = df.iloc[args.start_index:args.end_index] # select the target row ranges
    # Create Ouput Folder if not exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    df.parallel_apply(lambda row: get_row_image_embedding(args, row), axis=1)
    
    print("All done! Saved at path: %s"%(args.output_path))
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_retry", default=2, type=int, required=False, help = "choose number of times allowed to retry API call'")
    parser.add_argument("--img_dir", default="../datasets/get_ad_images_cr/creative_ranking/dataset_images/", type=str)
    parser.add_argument("--input_path", default="../datasets/get_ad_images_cr/all_ad_images.csv", type=str)
    parser.add_argument("--output_path", default="../datasets/get_ad_images_cr/preprocessed/image_embedding/", type=str)

    parser.add_argument("--nb_workers", type=int, default=4, help="Start Index of the batched processing")
    parser.add_argument("--start_index", type=int, default=0, help="Start Index of the batched processing")
    parser.add_argument("--end_index", type=int, default=0, help="End Index of the batched processing")
    
    args = parser.parse_args()
    print(args)
    
    main(args)
    