#!/bin/bash
# yes | pip install numpy
# yes | pip install pandas
# yes | pip install pandarallel
# yes | pip install joblib
# yes | pip install tqdm
yes | pip install bert-serving-server  # server
yes | pip install bert-serving-client # client

bert-serving-start -model_dir ../pretrained/chinese_L-12_H-768_A-12/ -num_worker=4
