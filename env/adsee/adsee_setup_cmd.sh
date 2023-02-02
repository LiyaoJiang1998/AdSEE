#!/bin/bash

# create conda environment for python
eval "$(conda shell.bash hook)"

conda create -y -n adsee python=3.7.16

# isntall ipykernel for using in ipynb
yes | pip install ipykernel
# create pythonkernel for jupyter launcher
python -m ipykernel install --user --name adsee --display-name "python (adsee)"

# activate the new env
conda activate adsee

# for using with jupyterlab
yes | pip install jupyterlab

conda install -y pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
yes | pip install ipykernel==6.12.2
yes | pip install matplotlib==3.5.3

# install SOLO
cd ../../SOLO
yes | pip install -r requirements/build.txt
# pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
cd ../env/adsee/cocoapi/PythonAPI
python setup.py build_ext install
cd ../../..
yes | pip install -v -e .  # or "python setup.py develop"
yes | pip install tqdm==4.64.1
yes | pip install ipywidgets==8.0.4
yes | pip install pandas==1.3.5
yes | pip install numpy==1.21.5
yes | pip install joblib==1.2.0
yes | pip install pandarallel==1.5.4
yes | pip install timm=0.6.12
yes | pip install imutils==0.5.4
# Merge/Add dependencies for CTR predictor and pygad:
yes | pip install tensorflow==2.0.0
yes | pip install protobuf==3.20.3
yes | pip install -U deepctr-torch==0.2.9
yes | pip install pygad==2.18.1
yes | pip install sklearn
yes | pip install img2vec_pytorch==1.0.1
yes | pip install opencv-python==4.2.0.34

conda update libstdcxx-ng

# psp_env dependencies
yes | pip install cmake==3.25.0
yes | pip install dlib==19.24.0
cd ../env/adsee/
# wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1

conda activate adsee
python -m ipykernel install --user --name adsee --display-name "python (adsee)"
conda deactivate
jupyter kernelspec list
