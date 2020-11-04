'''https://github.com/ilikewind/CAMELYON/blob/master/camelyon16/bin/tissue_mask.py'''

import os
import numpy as np
import openslide
import tifffile
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
import multiprocessing
import gc


def gen_tissue_mask(file_wsi, level, file_mask_npy, file_mask_tif, RGB_min=50):
    slide = openslide.OpenSlide(file_wsi)

    print(f'reading {file_wsi}...')
    img_RGB = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB'))
    print(f'reading {file_wsi} complete.')
    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    print(f'writing filename_mask_npy...')
    np.save(file_mask_npy, tissue_mask)
    print(f'writing filename_mask_npy complete.')

    img_mask = np.array(tissue_mask).astype(np.uint8)
    img_mask *= 255
    print(f'writing {file_mask_tif}...')
    tifffile.imwrite(file_mask_tif, img_mask, compress=9)
    print(f'writing {file_mask_tif} complete.')

    slide.close()
    del img_RGB, img_HSV, background_R, background_G, background_B, tissue_S, tissue_mask, img_mask
    gc.collect()


if __name__ == '__main__':
    level = 3
    wsi_dir = 'F:\\CAMELYON16\\training\\wsi\\'
    mask_dir = 'F:\\CAMELYON16\\training\\tissue_mask\\'
    RGB_min = 50

    multi_processes_num = 2
    pool = multiprocessing.Pool(processes=multi_processes_num)
    for dir_path, subpaths, files in os.walk(wsi_dir, False):
        for f in files:
            filename_wsi = os.path.join(dir_path, f)
            _, full_filename = os.path.split(filename_wsi)
            filename, file_extension = os.path.splitext(full_filename)
            if file_extension.upper() not in ['.TIF']:
                continue

            file_mask_npy = os.path.join(mask_dir, str(level), filename_wsi.replace(wsi_dir, '')).replace('.tif', '.npy')
            os.makedirs(os.path.dirname(file_mask_npy), exist_ok=True)
            file_mask_tif = os.path.join(mask_dir, str(level), filename_wsi.replace(wsi_dir, ''))
            pool.apply_async(gen_tissue_mask, (filename_wsi, level, file_mask_npy, file_mask_tif, RGB_min))

    pool.close()
    pool.join()

    print('OK')
