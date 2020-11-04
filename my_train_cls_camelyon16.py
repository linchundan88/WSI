import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from libs.Dataset.my_dataset import Dataset_CSV
from torch.utils.data import DataLoader
import torch.optim as optim
# import torch_optimizer as optim_plus
# from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import WeightedRandomSampler
from libs.neuralNetworks.classification.my_train_classification import train
import albumentations as A
import pandas as pd


#region prepare dataset

image_shape = (299, 299)

data_type = 'camelyon16'
data_version = 'v1'
filename_train = os.path.join(os.path.abspath('.'),
                'datafiles', f'{data_type}_{data_version}_train.csv')
filename_valid = os.path.join(os.path.abspath('.'),
                'datafiles', f'{data_type}_{data_version}_valid.csv')
filename_test = os.path.join(os.path.abspath('.'),
                'datafiles', f'{data_type}_{data_version}_test.csv')

transform_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(image_shape[0], image_shape[1]),  #(height,weight)
    ])


batch_size_train, batch_size_valid = 32, 64

df = pd.read_csv(filename_train)
num_class = df['labels'].nunique(dropna=True)

# list_class_samples = []
# for label in range(num_class):
#     list_class_samples.append(len(df[df['labels'] == label]))
# sample_class_weights = 1 / np.power(list_class_samples, 0.5)
'''
0 = {int} 280052
1 = {int} 53464
'''
sample_class_weights = [1, 3]
sample_weights = []
for label in df['labels']:
    sample_weights.append(sample_class_weights[label])
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(df))

ds_train = Dataset_CSV(csv_file=filename_train, transform=transform_train, image_shape=image_shape)
loader_train = DataLoader(ds_train, batch_size=batch_size_train,
                          sampler=sampler, num_workers=8, pin_memory=True)
ds_valid = Dataset_CSV(csv_file=filename_valid, image_shape=image_shape)
loader_valid = DataLoader(ds_valid, batch_size=batch_size_valid, num_workers=8, pin_memory=True)
ds_test = Dataset_CSV(csv_file=filename_test, image_shape=image_shape)
loader_test = DataLoader(ds_test, batch_size=batch_size_valid, num_workers=8, pin_memory=True)

#endregion

#region training
save_model_dir_base = '/home/stu/data_share/CAMELYON16/models/'

train_times = 1
# 'xception', 'inceptionresnetv2', 'inceptionv3'
for model_name in ['xception', 'inceptionresnetv2', 'inceptionv3']:
    import pretrainedmodels
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    num_filters = model.last_linear.in_features
    model.last_linear = nn.Linear(num_filters, num_class)

    loss_class_weights = [1, 2]
    loss_class_weights = torch.FloatTensor(loss_class_weights)
    if torch.cuda.is_available():
        loss_class_weights = loss_class_weights.cuda()
    label_smoothing = 0.1
    if label_smoothing > 0:
        from libs.neuralNetworks.classification.losses.my_label_smoothing import LabelSmoothLoss
        criterion = LabelSmoothLoss(class_weight=loss_class_weights, smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=loss_class_weights)

    epochs_num = 5

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    # optimizer = optim_plus.radam(model.parameters(), lr=0.001, weight_decay=0)
    # optimizer = optim_plus.Lookahead(optimizer, k=5, alpha=0.5)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs_num // 4, eta_min=0)  #T_max: half of one circle
    # optimizer = GradualWarmupScheduler(optimizer, multiplier=4, total_epoch=2, after_scheduler=scheduler)

    for i in range(train_times):
        save_model_dir = os.path.join(save_model_dir_base, data_type, data_version, str(i))
        train(model,
              loader_train=loader_train,
              criterion=criterion, optimizer=optimizer, scheduler=scheduler,
              epochs_num=epochs_num, log_interval_train=10,
              # loader_valid=loader_valid,
              loader_test=loader_test,
              save_model_dir=os.path.join(save_model_dir, model_name, str(i))
              )

#endregion

print('OK')


