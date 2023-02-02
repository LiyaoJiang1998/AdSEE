# %%
import joblib
import numpy as np
import pandas as pd
import os
import gc
from pandarallel import pandarallel

def load_pkl_row(path):
    try:
        result = joblib.load(path)
    except Exception as e:
        print("Exception:", e)
        print("Exception while loading:", path)
        result = {}
    return result

pandarallel.initialize()

# load pkls and convert to dfs
raw_df = pd.read_csv("all_ad_images.csv", sep='\t', header=0, index_col=0)
# raw_df = pd.read_csv("all_ad_images_nima.csv", sep='\t', header=0, index_col=0)

style_features_path = "preprocessed/e4e/e4e_"
image_features_path = "preprocessed/image_embedding/image_embedding_"
# text_features_bert_path = "preprocessed/text_embedding/bert_"

style_features = raw_df.parallel_apply(lambda row: load_pkl_row(style_features_path + 
                                    row["image_name"].replace(".jpg", ".pkl")), axis=1)
style_df = pd.DataFrame(style_features.tolist())
gc.collect()
del style_features
gc.collect()
print("style_features loading finished!")

image_features = raw_df.parallel_apply(lambda row: load_pkl_row(image_features_path + 
                                    row["image_name"].replace(".jpg", ".pkl")), axis=1)
image_df = pd.DataFrame(image_features.tolist())
gc.collect()
del image_features
gc.collect()
print("image_features loading finished!")

# text_features_bert = raw_df.parallel_apply(lambda row: load_pkl_row(text_features_bert_path + 
#                                     row["image_name"].replace(".jpg", ".pkl")), axis=1)
# text_bert_df = pd.DataFrame(text_features_bert.tolist())
# gc.collect()
# del text_features_bert
# gc.collect()
# print("text_features_bert loading finished!")

# %%
# import joblib
# import numpy as np
# import pandas as pd
# import os
# import gc

# raw_df = pd.read_csv("preprocessed/all_ad_images_nima.csv", sep='\t', header=0, index_col=0)
# train_df = joblib.load("preprocessed/train_df_with_no_face.pkl")

# train_df["nima_mean"] = raw_df["nima_mean"]
# train_df["nima_std"] = raw_df["nima_std"]
# train_df = train_df[(train_df["nima_mean"] != 0) | (train_df["nima_std"] != 0)]
# joblib.dump(train_df, "preprocessed/train_df_with_no_face.pkl")

# %%
'''
Run when e4e is not finished, but only solo is finished.
total image: 
Number of images with person: 
Ratio of person with Images: 
'''
# solo_features_path = "preprocessed/solo/solo_"
# solo_features = raw_df.parallel_apply(lambda row: joblib.load(solo_features_path + 
#                                     row["image_name"].replace(".jpg", ".pkl")), axis=1)
# print("solo_features loading finished!")
# solo_df = pd.DataFrame(solo_features.tolist())

# print(len(solo_df))
# # images with at least one person:
# num_person_images = (solo_df['person_count'] > 0).sum()
# print("Number of images with person:", num_person_images)
# print("Ratio of person with Images:", num_person_images / len(solo_df))

# %%
# print some column names and count
print("Number of rows should be same:", len(raw_df), len(style_df), len(image_df))
print(raw_df.columns)
print(style_df.columns)
print(image_df.columns)
# print(text_bert_df.columns)

# invalid images:
num_invalid_images = (style_df['valid_image'] == False).sum()
print("Number of invalid images:", num_invalid_images)

# # invalid texts:
# num_invalid_texts = ((text_bert_df['valid_text'] == False)).sum()
# print("Number of invalid texts:", num_invalid_texts)

# images with at least one person:
num_person_images = (style_df['person_count'] > 0).sum()
print("Number of images with person:", num_person_images)
print("Ratio of person with Images:", num_person_images / len(style_df))

# images with at least one face:
num_face_images = (style_df['face_count'] > 0).sum()
print("Number of images with face:", num_face_images)
print("Ratio of Images with face:", num_face_images / len(style_df))

# %%
# Rename some columns
raw_df = raw_df.rename({"(total_click / total_show)":'ctr'}, axis=1)
# text_bert_df = text_bert_df.rename({'name_embedding':'name_embedding_bert'}, axis=1)

# Merge dfs
# text_bert_df[["name_embedding_bert"]],
merged_df = pd.concat(
            [raw_df[["image_name", "product_name",
                     "image_url", "total_click", "total_show", "ctr"]],
            style_df[["valid_image", "person_count", "person_masks",
                      "category_labels", "category_name_labels",
                      "face_count", "face_latents_index", "face_latents"]],
            image_df[["img_embedding"]],
        ], axis=1, join='inner')

# Filter out rows with 0 ctr, because log(0) is -inf
merged_df = merged_df[merged_df["ctr"] != 0]
# Filter out any row with na values
merged_df = merged_df.dropna()
# Filter out any rows with NIMA mean and std both equal zeros
# merged_df = merged_df[(merged_df["nima_mean"] != 0) | (merged_df["nima_std"] != 0)]

# Change img_embedding to np.arrays instead of list
merged_df["img_embedding"] = merged_df["img_embedding"].apply(lambda x: np.array(x, dtype=np.float32))


# CTR is left skewed, take log
merged_df.insert(loc=6, column='log_ctr', value=np.log(merged_df["ctr"]))
# Standardize log CTR to overall mean and std
s_log_ctr = (merged_df["log_ctr"]-merged_df["log_ctr"].mean())/merged_df["log_ctr"].std()
log_ctr_mean = merged_df["log_ctr"].mean()
log_ctr_std = merged_df["log_ctr"].std()
print(log_ctr_mean, log_ctr_std)
merged_df.insert(loc=7, column='s_log_ctr', value=s_log_ctr)

# Filter out invalid columns
max_num_person = 5
# filtered_df = merged_df[(merged_df["valid_image"] == True) &
#                 (merged_df['face_count'] > 0) & 
#                 (merged_df['face_count'] <= max_num_person)]
filtered_df = merged_df[(merged_df["valid_image"] == True) & 
                (merged_df['face_count'] <= max_num_person)]

# Flatten face_latents, and pad to 5 person* 18*512..., to make it fixed length
style_vector_dim = (18, 512)
final_size = max_num_person * style_vector_dim[0] * style_vector_dim[1]

# Flatten:
filtered_df["face_latents"] = filtered_df["face_latents"].apply(lambda x : np.array(x, dtype=np.float32).flatten())
# # Zero-Padding:
# filtered_df["face_latents"] = filtered_df["face_latents"].apply(lambda x : 
#                 np.array(np.pad(x, (0, final_size - x.size), mode='constant', constant_values=0), dtype=np.float32))

print(filtered_df["face_count"].value_counts())

# Prepare only the columns used for training and editing
#                        'name_embedding_bert',
train_df = filtered_df[["image_name", "product_name", "ctr", "log_ctr", "s_log_ctr",
                        "category_labels", "category_name_labels",
                        "face_count", "face_latents", 
                        "img_embedding",
                        "face_latents_index"]]



# %%
import gc
gc.collect()

del raw_df
del style_df
del image_df
# del text_bert_df

gc.collect()

# %%
print(merged_df.columns)
print(train_df.columns)


# %%
# # Saving full and filtered dfs
# joblib.dump(train_df, "preprocessed/train_df.pkl")
joblib.dump(train_df, "preprocessed/train_df_with_no_face.pkl")

log_ctr_mean = merged_df["log_ctr"].mean()
log_ctr_std = merged_df["log_ctr"].std()
print(log_ctr_mean, log_ctr_std)

# %%
# Some Analysis
# print(filtered_df[filtered_df["face_count"] >0]['product_name'].value_counts())
print(merged_df["face_count"].value_counts())

# %%
print(train_df.corr())

# %%
joblib.dump(merged_df, "preprocessed/merged_df.pkl")
joblib.dump(filtered_df, "preprocessed/filtered_df.pkl")
