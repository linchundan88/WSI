import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from libs.neuralNetworks.classification.my_predict import predict_csv

image_shape = (299, 299)
num_class = 2
filename_csv = '/home/stu/data_share/PAIP2020/predict/patches_validate.csv'
model_file = '/home/stu/data_share/paip2000/models/paip2000/level0_299_299/0/xception/0/epoch2.pth'

import pretrainedmodels
model_name = 'xception'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
num_filters = model.last_linear.in_features
model.last_linear = nn.Linear(num_filters, num_class)
state_dict = torch.load(model_file, map_location='cpu')
model.load_state_dict(state_dict)

probs = predict_csv(model, filename_csv, image_shape, model_convert_gpu=True, batch_size=64, argmax=False)
preds = probs.argmax(axis=-1)

import numpy as np
file_file = '/home/stu/data_share/PAIP2020/predict/probs_validate.npy'
np.save(file_file, probs)

import pandas as pd
df = pd.read_csv(filename_csv)
labels = df['labels']
from sklearn.metrics import confusion_matrix
print('Confusion Matrix:', confusion_matrix(list(labels), list(preds)))


print('OK')

