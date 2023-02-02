import time
import os
import requests
import pandas as pd
import shutil
import PIL.Image

from pandarallel import pandarallel


# def DownloadImage(img_url, save_name, save_dir, num_retry=100):
#     if (type(img_url) != str) or ('http' not in img_url) or (os.path.exists(os.path.join(save_dir, save_name))):    
#         return
    
#     for i in range(0, num_retry):        
#         try:
#             img_data = requests.get(img_url).content
#             with open(os.path.join(save_dir, save_name), 'wb') as handler:
#                 handler.write(img_data)
#             return
#         except Exception as e:
#             time.sleep(0.01)
#             continue
    
#     raise ValueError("Exceeding Retry Limit for image %s, %s"%(save_name, img_url))


def MoveImage(img_url, img_dir, save_name, save_dir, num_retry=5):
    if (type(img_url) != str) or (os.path.exists(os.path.join(save_dir, save_name))):    
        return
    
    for i in range(0, num_retry):        
        try:
            src, dst = os.path.join(img_dir, img_url), os.path.join(save_dir, save_name)
            img = PIL.Image.open(src)
            img.save(dst)
            return
        except Exception as e:
            time.sleep(0.01)
            continue
    
    raise ValueError("Exceeding Retry Limit for image %s, %s"%(save_name, img_url))


if __name__ == '__main__':
    pandarallel.initialize(progress_bar=True)
    
    # load from data list of CreativeRanking Dataset
    # product name, image name, displayed date, number of impressions and number of clicks
    df1 = pd.read_csv('creative_ranking/list/train_data_list.txt', sep='\t', header=None)
    df2 = pd.read_csv('creative_ranking/list/val_data_list.txt', sep='\t', header=None)
    df3 = pd.read_csv('creative_ranking/list/test_data_list.txt', sep='\t', header=None)
    
    # df = pd.concat([df1,df2,df3])
    df = df1
    df.columns = ["product_name",
                  "image_url",
                  "display_date",
                  "total_show",
                  "total_click"]
    
    # group the same ad and same image for all different days.
    df = df.groupby(by=['product_name','image_url'])[["total_show", "total_click"]].sum().reset_index()
    
    # Sort and Filter
    df["(total_click / total_show)"] = df["total_click"] / df["total_show"]
    df = df.sort_values(by=["(total_click / total_show)", "total_show", "image_url"], ascending=False) # Sort by CTR
    df = df[df["total_click"] > 0]
    df = df[df["total_show"] > 100]
    df = df[df["total_show"] < 1000]
    df = df.dropna().reset_index(drop=True) # drop NA rows
    
    df.insert(0, column="id", value=df.index.values)
    df.insert(1, column="image_name", value=df.index.values)
    df['image_name'] = df['image_name'].astype(str) + ".jpg"

    new_col_order = ['id', 'image_name', 'product_name', 'image_url', \
                     'total_click', 'total_show', "(total_click / total_show)"]
    df = df.reindex(new_col_order, axis=1)
        
    # save csv
    df.to_csv("all_ad_images.csv", sep='\t', header='True', index=False)
    
    print(df.describe())
    print(df.head())
    print(df.tail())
    
    # move images （img_url, img_dir, save_name, save_dir）
    df.parallel_apply(lambda row: MoveImage(row['image_url'], "./creative_ranking/images/", row['image_name'], "./creative_ranking/dataset_images/"), axis=1)
