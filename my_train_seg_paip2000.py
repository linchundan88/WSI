import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from libs.neuralNetworks.semanticSegmentation.models.u_net import Unet
from libs.dataset.my_dataset import Dataset_CSV_sem_seg
from torch.utils.data import DataLoader
import torch.optim as optim
# import torch_optimizer as optim_plus
from torch.optim.lr_scheduler import StepLR
# from warmup_scheduler import GradualWarmupScheduler
from libs.neuralNetworks.semanticSegmentation.my_train_sem_seg import train
import albumentations as A


#region prepare dataset
data_type = 'paip2000_seg'
data_version = 'wsi_level0_384_384__num4000'
image_shape = (384, 384)
filename_train = os.path.join(os.path.abspath('.'),
                'datafiles', f'{data_type}_{data_version}_train.csv')
filename_valid = os.path.join(os.path.abspath('.'),
                'datafiles', f'{data_type}_{data_version}_valid.csv')
filename_test = os.path.join(os.path.abspath('.'),
                'datafiles', f'{data_type}_{data_version}_test.csv')

transform_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # ToTensor()
    ])

batch_size_train, batch_size_valid = 16, 32

ds_train = Dataset_CSV_sem_seg(csv_file=filename_train, transform=transform_train, image_shape=image_shape)
loader_train = DataLoader(ds_train, batch_size=batch_size_train,
                          shuffle=True,  num_workers=8, pin_memory=True)
ds_valid = Dataset_CSV_sem_seg(csv_file=filename_valid, image_shape=image_shape)
loader_valid = DataLoader(ds_valid, batch_size=batch_size_valid, num_workers=8, pin_memory=True)
ds_test = Dataset_CSV_sem_seg(csv_file=filename_test, image_shape=image_shape)
loader_test = DataLoader(ds_test, batch_size=batch_size_valid, num_workers=8, pin_memory=True)

#endregion

#region training
model_name = 'Unet'
save_model_dir_base = '/home/stu/data_share/paip2000/models_seg/'

train_times = 1
for i in range(train_times):
    import segmentation_models_pytorch as smp
    # model = smp.Unet()
    model = smp.Unet('resnet18', encoder_weights='imagenet', in_channels=3, encoder_depth=4, decoder_channels=4, activation=None)
    # model = Unet(in_ch=3, out_ch=1, channel_nums=[64, 128, 256, 512, 1024], activation=None)

    pos_weight = torch.FloatTensor(torch.tensor([3.]))
    if torch.cuda.is_available():
        pos_weight = pos_weight.cuda()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # from libs.neuralNetworks.semanticSegmentation.losses.my_loss_sem_seg import diceCoeff, DiceLoss
    # criterion = DiceLoss(activation='sigmoid')

    epochs_num = 5

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)
    # optimizer = optim_plus.radam(model.parameters(), lr=0.001, weight_decay=0)
    # optimizer = optim_plus.Lookahead(optimizer, k=5, alpha=0.5)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs_num // 4, eta_min=0)  #T_max: half of one circle
    # scheduler = GradualWarmupScheduler(optimizer, multiplier=4, total_epoch=2, after_scheduler=scheduler)

    save_model_dir = os.path.join(save_model_dir_base, data_type, data_version, str(i))
    train(model,
          loader_train=loader_train,
          criterion=criterion, optimizer=optimizer, scheduler=scheduler,
          epochs_num=epochs_num, log_interval_train=10,
          loader_valid=loader_valid, loader_test=loader_test,
          save_model_dir=os.path.join(save_model_dir, model_name, str(i))
          )

#endregion

print('OK')

