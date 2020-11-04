import numpy as np
import openslide
import tifffile
import pandas as pd
import cv2


filename_wsi = '/home/stu/data_share/PAIP2020/wsi/training_data_01.svs'
wsi_level = 0  # the WSI level that extract patches from


slide = openslide.OpenSlide(filename_wsi)
(width, height) = slide.level_dimensions[0]
img_gt = np.zeros([height, width], dtype=np.uint8)
img_predict = np.zeros([height, width], dtype=np.uint8)

file_file = '/home/stu/data_share/PAIP2020/predict/probs_validate.npy'
probs = np.load(file_file)
predicts = np.argmax(probs, axis=1)

filename_csv = '/home/stu/data_share/PAIP2020/predict/patches_validate.csv'
df = pd.read_csv(filename_csv)
for index, row in df.iterrows():
    image_file, label = row['images'], row['labels']
    (y1, y2, x1, x2) = row['y1'], row['y2'], row['x1'], row['x2']
    print((x1, y1, x2, y2))

    if label == 1:
        cv2.rectangle(img_gt, (x1, y1), (x2, y2), (255, 255, 255), -1)
    if predicts[index] == 1:
        cv2.rectangle(img_predict, (x1, y1), (x2, y2), (255, 255, 255), -1)


img_gt = cv2.resize(img_gt, (slide.level_dimensions[2]))
img_predict = cv2.resize(img_predict, (slide.level_dimensions[2]))

print(np.sum(img_gt > 0))
file_result_gt = '/home/stu/data_share/PAIP2020/predict/result_gt.tif'
file_result_predict = '/home/stu/data_share/PAIP2020/predict/result_predict.tif'
import time
print(time.time())
tifffile.imwrite(file_result_gt, img_gt, compress=9)
tifffile.imwrite(file_result_predict, img_predict, compress=9)
print(time.time())

print('OK')