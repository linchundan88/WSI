'''
Generating Tissue masks with a threshold value-RGB(235, 210, 235)- for each original whole slide image;
the pixel with a value above the threshold is considered as 'background'(non-tissue area), vice versa.
'''


import os
import numpy as np
import openslide
import tifffile
from skimage.filters import threshold_otsu

RGB_max = (235, 210, 235)

wsi_dir = '/home/stu/data_share/PAIP2020/wsi/'
tissue_dir = '/home/stu/data_share/PAIP2020/tissue_mask/'
os.makedirs(tissue_dir, exist_ok=True)

level = 2  #training_data_22_l2_annotation_tumor.tif
for dir_path, subpaths, files in os.walk(wsi_dir, False):
    for f in files:
        file_wsi = os.path.join(dir_path, f)
        _, filename = os.path.split(file_wsi)
        filename_base, file_ext = os.path.splitext(filename)
        if file_ext.upper() not in ['.SVS']:
            continue

        slide = openslide.OpenSlide(file_wsi)
        (width, height) = slide.level_dimensions[level]

        img_RGB = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB'))

        # background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
        # background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
        # background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])

        min_R = img_RGB[:, :, 0] < RGB_max[0]
        min_G = img_RGB[:, :, 1] < RGB_max[1]
        min_B = img_RGB[:, :, 2] < RGB_max[2]

        tissue_mask = min_R & min_G & min_B

        img_mask = np.array(tissue_mask).astype(np.uint8)
        img_mask *= 255

        file_mask_tif = os.path.join(tissue_dir, filename_base + '_l2_annotation_tissue.tif')
        print(f'writing {file_mask_tif}...')
        tifffile.imwrite(file_mask_tif, img_mask, compress=9)
        print(f'writing {file_mask_tif} complete.')

print('OK')




