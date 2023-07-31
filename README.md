# AdSEE: Investigating the Impact of Image Style Editing on Advertisement Attractiveness (KDD 2023)
| [Paper](https://raw.githubusercontent.com/LiyaoJiang1998/AdSEE/main/adsee_kdd2023.pdf) |

<details>
<summary><strong>Click here to show the abstract </strong></summary>
Online advertisements are important elements in e-commerce sites, social media platforms, and search engines. With the increasing popularity of mobile browsing, many online ads are displayed with visual information in the form of a cover image in addition to text descriptions to grab the attention of users. Various recent studies have focused on predicting the click rates of online advertisements aware of visual features or composing optimal advertisement elements to enhance visibility. In this paper, we propose Advertisement Style Editing and Attractiveness Enhancement (AdSEE), which explores whether semantic editing to ads images can affect or alter the popularity of online advertisements. We introduce StyleGAN-based facial semantic editing and inversion to ads images and train a click rate predictor attributing GAN-based face latent representations in addition to traditional visual and textual features to click rates. Through a large collected dataset named QQ-AD, containing 20,527 online ads, we perform extensive offline tests to study how different semantic directions and their edit coefficients may impact click rates. We further design a Genetic Advertisement Editor to efficiently search for the optimal edit directions and intensity given an input ad cover image to enhance its projected click rates. Online A/B tests performed over a period of 5 days have verified the increased click-through rates of AdSEE-edited samples as compared to a control group of original ads, verifying the relation between image styles and ad popularity. We open source the code for AdSEE research at https://github.com/LiyaoJiang1998/adsee.
</details>
</details>

## Description
This repo supports the KDD '23 paper ["AdSEE: Investigating the Impact of Image Style Editing on Advertisement Attractiveness"](https://raw.githubusercontent.com/LiyaoJiang1998/AdSEE/main/adsee_kdd2023.pdf). It contains the official implementaion for the AdSEE framework inlcuding a Click Rate Predictor (CRP) and a Genetic Advertisement Editor (GADE).

## Environment Setup
* To run this code, we use a Anaconda Python Virtual Environment to manage and install the dependencies. Please install it from <https://www.anaconda.com/>
* We use Python 3.7.16, PyTorch 1.7.0, Nvidia GPU, and Linux. The other dependencies are in `env/adsee/adsee_setup_cmd.sh`
* Use the following commnads to install and activate the enviroment
    ```
    git clone https://github.com/LiyaoJiang1998/AdSEE
    cd env/adsee/
    sh adsee_setup_cmd.sh
    conda deactivate
    conda activate adsee
    ```

## Pre-trained Models
* Our code adopts the following pre-trained models, please download the pre-trained models from the following links and save in specified directories.

    * SOLO Instance Segmentation Models
        * Download Link: <https://cloudstor.aarnet.edu.au/plus/s/BRhKBimVmdFDI9o/download>
        * Destination: `adsee/SOLO/checkpoints/DECOUPLED_SOLO_R101_3x.pth`
        * Download Link: <https://cloudstor.aarnet.edu.au/plus/s/4ePTr9mQeOpw0RZ/download>
        * Destination: `adsee/SOLO/checkpoints/SOLOv2_R101_DCN_3x.pth`
    * encoder4editing FFHQ Encoder
        * Download Link: <https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view>
        * Destinations:
            * `encoder4editing/pretrained_models/e4e_ffhq_encode.pt`
            * `ctr_predictor/checkpoints/e4e_ffhq_encode.pt`
    * Dlib Face Alignment Model
        * Download Instrutions:
            ```
            wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
            bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
            cp shape_predictor_68_face_landmarks.dat adsee/encoder4editing/
            cp shape_predictor_68_face_landmarks.dat ctr_predictor/checkpoints/
            ```
        * Destinations:
            * `encoder4editing/shape_predictor_68_face_landmarks.dat`
            * `ctr_predictor/checkpoints/shape_predictor_68_face_landmarks.dat`
    * Bert Chinese Model
        * Download Instructions:
            ```
            wget https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip -o text_embedding/pretrained/
            cd text_embedding/pretrained/; unzip multilingual_L-12_H-768_A-12.zip
            ```
        * Destination: `text_embedding/pretrained/chinese_L-12_H-768_A-12/`

## Dataset
* We use the CreativeRanking Dataset, please dowload from: <https://tianchi.aliyun.com/dataset/93585>
* Put the dataset images and list, under `AdSEE/datasets/get_ad_images_cr/creative_ranking/images` and `AdSEE/datasets/get_ad_images_cr/creative_ranking/list`
* Dataset is from paper:
  > @inproceedings{wang2021hybrid,
  >              title={A Hybrid Bandit Model with Visual Priors for Creative Ranking in Display Advertising},
  >              author={Wang, Shiyao and Liu, Qi and Ge, Tiezheng and Lian, Defu and Zhang, Zhiqiang},
  >              booktitle={Proceedings of the 30th international conference on World wide web},
  >              year={2021}}

## Training and Testing of Click Rate Predictor
1. SOLO instance segmentation preprocessing
    * Obtain the human segmentation masks and labels of other object classes using SOLO isntance segmentation model for images from the CreativeRanking dataset.
    * Use the following command:
        ```
        cd SOLO/
        CUDA_VISIBLE_DEVICES=0 python solo_preprocess_batched.py --start_index 0 --end_index 133681
        CUDA_VISIBLE_DEVICES=1 python solo_preprocess_batched.py --start_index 133681 --end_index 267362
        ```

2. e4e encoder preprocessing
    * Obtain the face latent codes by first aligning the faces with Dlib, and extract latent codes from e4e FFHQ encoder for images from the CreativeRanking dataset.
    * Use the following command:
        ```
        cd encoder4editing/
        CUDA_VISIBLE_DEVICES=0 python preprocessing_face_latent_batched.py --start_index 0 --end_index 133681
        CUDA_VISIBLE_DEVICES=1 python preprocessing_face_latent_batched.py --start_index 133681 --end_index 267362
        ```

3. Image Embedding
    * Obtain the image embedding of each image from the CreativeRanking dataset by using the ResNet-18 model in the img2vec library. 
    * Use the following command:
        ```
        cd image_embedding/
        CUDA_VISIBLE_DEVICES=0 python image_embedding_preprocessing.py --start_index 0 --end_index 133681
        CUDA_VISIBLE_DEVICES=1 python image_embedding_preprocessing.py --start_index 133681 --end_index 267362
        ```

4. Text Embedding
    * (Skip) Since CreativeRanking does not contain text data, text emdedding is disabled for CreativeRanking dataset.
    * If your own dataset have text information, please extract the text embedding as follows:
        ```
        # In one terminal，run bert model server 
        cd text_embedding/server/
        sh starup.sh
        # In another terminal，run bert model client
        cd text_embedding/
        python text_embedding_bert.py
        ```

5. Combining Preprocessed Dataset
    * Merging the results from previous steps into a single `.pkl` data file.
    * Use the following command:
        ```
        cd datasets/get_ad_images_cr/
        python combine_preprocessed.py
        ```

6. Training of Click Rate Predictor
    * We train a Click Rate Predictor model on the preprocessed CreativeRanking dataset.
    * The Click Rate Predictor is trained and tested in the following scirpt:
        ```
        cd adsee/ctr_predictor
        CUDA_VISIBLE_DEVICES=0 python train_predictor.py --learning_rate 0.00001 --exp_name="ctr_predictor_default" --ctr_model="AutoInt" --target="s_log_ctr" --dense_array_features='["face_latents", "img_embedding"]' --num_epoch 18 --style_vector_method="max_pooling"  --val_split 0.2 --test_split 0.2
        ```

## Image Editing with GADE module

* In this section, we use AdSEE including the GADE module and the Click Rate Predictor trained in the previous section to enhance and edit images from the CreativeRanking dataset guided by the Click Rate Predictor.
* Please use the following command to use AdSEE to edit the images:
    ```
    cd ctr_predictor

    # To Edit all images from the test set:
    # CUDA_VISIBLE_DEVICES=0 python GA_optimize.py --ga_exp_name ga_default_test_data_full --num_processes 4

    # To Edit 500 images sampled from the test set:
    CUDA_VISIBLE_DEVICES=0 python GA_optimize.py --ga_exp_name ga_default_test_data_sample_500_start_0_end_250 --num_processes 4 --edit_samples 500 --start_index 0 --end_index=250
    CUDA_VISIBLE_DEVICES=1 python GA_optimize.py --ga_exp_name ga_default_test_data_sample_500_start_250_end_500 --num_processes 4 --edit_samples 500 --start_index 250 --end_index=500
    ```
* The edited images will be saved at `ctr_predictor/results/ga_default_test_data_sample_500_start_0_end_250/edited/`

## Acknowledgments
We want to acknowledge that our implementation adopt code from the following repositories, and we thank the authors for sharing their code:
* [omertov/encoder4editing](https://github.com/omertov/encoder4editing/)
* [genforce/sefa](https://github.com/genforce/sefa)
* [WXinlong/SOLO](https://github.com/WXinlong/SOLO)
* [ahmedfgad/GeneticAlgorithmPython](https://github.com/ahmedfgad/GeneticAlgorithmPython)
* [christiansafka/img2vec](https://github.com/christiansafka/img2vec)
* [jina-ai/clip-as-service](https://github.com/jina-ai/clip-as-service/)
* [google-research/bert](https://github.com/google-research/bert)

## Citation

If you find this research useful, please cite our paper.
```
@InProceedings{jiang2023adsee,
  author       = {Liyao Jiang and Chenglin Li and Haolan Chen and Xiaodong Gao and Xinwang Zhong and Yang Qiu and Shani Ye and Di Niu},
  booktitle    = {Proceedings of the 29th {ACM} {SIGKDD} Conference on Knowledge Discovery and Data Mining},
  title        = {{AdSEE}: Investigating the Impact of Image Style Editing on Advertisement Attractiveness},
  year         = {2023},
  month        = {aug},
  publisher    = {{ACM}},
  doi          = {10.1145/3580305.3599770},
}
```