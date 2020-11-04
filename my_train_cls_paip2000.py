import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from libs.dataset.my_dataset import Dataset_CSV
from torch.utils.data import DataLoader
import torch.optim as optim
# import torch_optimizer as optim_plus
from torch.optim.lr_scheduler import StepLR
# from warmup_scheduler import GradualWarmupScheduler
from libs.neuralNetworks.classification.my_train_classification import train
import albumentations as A
import pandas as pd
import pretrainedmodels


#region prepare dataset

image_shape = (299, 299)

data_type = 'paip2000'
data_version = 'level0_299_299'
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
        # ToTensor()
    ])


batch_size_train, batch_size_valid = 64, 64

df = pd.read_csv(filename_train)
num_class = df['labels'].nunique(dropna=True)

# list_class_samples = []
# for label in range(num_class):
#     list_class_samples.append(len(df[df['labels'] == label]))
# sample_class_weights = 1 / np.power(list_class_samples, 0.5)

# sample_class_weights = [1, 3]
# sample_weights = []
# for label in df['labels']:
#     sample_weights.append(sample_class_weights[label])
# sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(df))

ds_train = Dataset_CSV(csv_file=filename_train, transform=transform_train, image_shape=image_shape)
loader_train = DataLoader(ds_train, batch_size=batch_size_train,
                          shuffle=True,  num_workers=8, pin_memory=True)
ds_valid = Dataset_CSV(csv_file=filename_valid, image_shape=image_shape)
loader_valid = DataLoader(ds_valid, batch_size=batch_size_valid, num_workers=8, pin_memory=True)
ds_test = Dataset_CSV(csv_file=filename_test, image_shape=image_shape)
loader_test = DataLoader(ds_test, batch_size=batch_size_valid, num_workers=8, pin_memory=True)

'''
paip2000_test_level0_299_299
2175606
0 1426721
1 748885
316839
0 215435
1 101404
207583
0 135830
1 71753

paip2000_test_level1_299_299
140158
0 92414
1 47744
20303
0 13872
1 6431
13688
0 9040
1 4648


/home/stu/data_share/paip2000/models/paip2000/level0_299_299/0/xception/0/epoch2.pth
[[1400233, 26488]
[19691, 729194]]

[[120243,15587]
[2137, 696161]]

'''

#endregion

#region training
save_model_dir_base = '/home/stu/data_share/paip2000/models/'

train_times = 1
# 'xception', 'inceptionresnetv2', 'inceptionv3'

for i in range(train_times):
    for model_name in ['xception', 'inceptionresnetv2', 'inceptionv3']:
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        num_filters = model.last_linear.in_features
        model.last_linear = nn.Linear(num_filters, num_class)

        loss_class_weights = [1., 1.3]
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
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
        # optimizer = optim_plus.radam(model.parameters(), lr=0.001, weight_decay=0)
        # optimizer = optim_plus.Lookahead(optimizer, k=5, alpha=0.5)

        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        # scheduler = CosineAnnealingLR(optimizer, T_max=epochs_num // 4, eta_min=0)  # T_max: half of one circle
        # optimizer = GradualWarmupScheduler(optimizer, multiplier=4, total_epoch=2, after_scheduler=scheduler)

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

