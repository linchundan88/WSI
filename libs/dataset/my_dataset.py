
import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from torchvision import transforms
import numpy as np


class Dataset_CSV(Dataset):
    def __init__(self, csv_file, transform=None, image_shape=None, test_mode=False):
        assert os.path.exists(csv_file), 'csv file does not exists'
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        assert len(self.df) > 0, 'csv file is empty!'
        self.image_shape = image_shape
        self.transform = transform
        self.test_mode = test_mode

    def __getitem__(self, index):
        file_img = self.df.iloc[index][0]
        assert os.path.exists(file_img), 'image file does not exists'
        image = cv2.imread(file_img)
        assert image is not None, f'{file_img} error.'
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #shape: height, width,  resize: width, height  #use A.Resize(image_shape) instead.
        # if (self.image_shape is not None) and (image.shape[:2] != self.image_shape[:2]):
        #     image = cv2.resize(image, (self.image_shape[1], self.image_shape[0]))

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        image = transforms.ToTensor()(image)

        if self.test_mode:
            return image
        else:
            label = int(self.df.iloc[index][1])
            return image, label

    def __len__(self):
        return len(self.df)


class Dataset_CSV_sem_seg(Dataset):
    def __init__(self, csv_file, transform=None, image_shape=None, test_mode=False):
        assert os.path.exists(csv_file), 'csv file does not exists'
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        assert len(self.df) > 0, 'csv file is empty!'
        self.image_shape = image_shape
        self.transform = transform
        self.test_mode = test_mode

    def __getitem__(self, index):
        file_img = self.df.iloc[index][0]
        assert os.path.exists(file_img), 'image file does not exists'
        image = cv2.imread(file_img)
        assert image is not None, f'{file_img} error.'
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        file_mask = self.df.iloc[index][1]
        assert os.path.exists(file_mask), 'image file does not exists'
        mask = cv2.imread(file_mask, cv2.IMREAD_GRAYSCALE)
        assert mask is not None, f'{file_mask} error.'

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)  # >127 set to 255, else 0. 127 set to 1,
        mask = np.expand_dims(mask, axis=0)

        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask)

        if self.test_mode:
            return image
        else:
            return image, mask

    def __len__(self):
        return len(self.df)


'''

# image2 = A.RandomRotate90(p=1)(image=image)['image']  #albumentations
        
from imgaug import augmenters as iaa
imgaug_iaa = iaa.Sequential([
    # iaa.CropAndPad(percent=(-0.04, 0.04)),
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.Flipud(0.2),  # horizontally flip 50% of the images

    # iaa.GaussianBlur(sigma=(0.0, 0.3)),
    # iaa.MultiplyBrightness(mul=(0.8, 1.2)),
    # iaa.contrast.LinearContrast((0.8, 1.2)),
    # iaa.Sometimes(0.9, iaa.Add((-8, 8))),
    # iaa.Sometimes(0.9, iaa.Affine(
    #     scale=(0.98, 1.02),
    #     translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
    #     rotate=(-15, 15),  # rotate by -10 to +10 degrees
    # )),
])
'''