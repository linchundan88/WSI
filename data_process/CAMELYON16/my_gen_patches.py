
import os
import cv2
import numpy as np
import random
import uuid
import multiprocessing
from libs.dataPreprocess.my_patches import gen_patches_classification, gen_patches_segmentation



if __name__ == '__main__':

    wsi_level = 0
    tissue_level = 4
    tumor_level = 4

    wsi_dir = '/home/stu/data_share/CAMELYON16/training/wsi'
    tissue_mask_dir = f'/home/stu/data_share/CAMELYON16/training/tissue_mask/{tissue_level}/'
    tumor_mask_dir = f'/home/stu/data_share/CAMELYON16/training/tumor_mask/{tumor_level}/'
    patches_dir_base = '/home/stu/data_share/CAMELYON16/training/patches/classification/'
    # patches_dir_base = '/home/stu/data_share/CAMELYON16/training/patches/segmentation/'


    patch_h, patch_w = (320, 320)

    sample_num_normal = 3000
    sample_num_tumor_normal = 2000
    sample_num_tumor = 1000

    tissue_threshold = 10
    tumor_threshold_high = 2
    tumor_threshold_lower = 0.1

    multi_processes_num = 8
    pool = multiprocessing.Pool(processes=multi_processes_num)

    for dir_path, subpaths, files in os.walk(wsi_dir, False):
        for f in files:
            filename_wsi = os.path.join(dir_path, f)
            _, filename = os.path.split(filename_wsi)
            filename_base, file_extension = os.path.splitext(filename)

            if filename_base.startswith('normal_'):
                file_tissue = filename_wsi.replace(wsi_dir, tissue_mask_dir)
                patches_dir = os.path.join(patches_dir_base, '0', filename_base)
                gen_grid_patches = True
                random_patch_ratio = None
                pool.apply_async(gen_patches_classification, (filename_wsi, wsi_level, patch_h, patch_w,
                                                              file_tissue, tissue_level, tissue_threshold,
                                                              gen_grid_patches, random_patch_ratio, patches_dir,
                                                              sample_num_normal))
                # pool.apply_async(gen_patches_segmentation, (filename_wsi, wsi_level, patch_h, patch_w,
                #                                       file_tissue, tissue_level, tissue_threshold,
                #                                     gen_grid_patches, random_patch_ratio, patches_dir,
                #                                      sample_num_normal, False))

            if filename_base.startswith('tumor_'):
                file_tissue = filename_wsi.replace(wsi_dir, tissue_mask_dir)
                file_tumor = filename_wsi.replace(wsi_dir, tumor_mask_dir).replace('/tumor/', '')

                patches_dir = os.path.join(patches_dir_base, '0', filename_base)
                gen_grid_patches = True
                random_patch_ratio = None
                pool.apply_async(gen_patches_classification, (filename_wsi, wsi_level, patch_h, patch_w,
                                                              file_tissue, tissue_level, tissue_threshold,
                                                              gen_grid_patches, random_patch_ratio, patches_dir,
                                                              sample_num_tumor_normal,
                                                              file_tumor, tumor_level, tumor_threshold_lower))

                # pool.apply(gen_patches_segmentation, (filename_wsi, wsi_level, patch_h, patch_w,
                #                                     file_tissue, tissue_level, tissue_threshold,
                #                                     gen_grid_patches, random_patch_ratio, patches_dir,
                #                                     sample_num_tumor_normal, False,
                #                                     file_tumor, tumor_level, tumor_threshold_lower ))

                patches_dir = os.path.join(patches_dir_base, '1', filename_base)
                gen_grid_patches = True
                random_patch_ratio = 2
                pool.apply_async(gen_patches_classification, (filename_wsi, wsi_level, patch_h, patch_w,
                                                              file_tumor, tumor_level, tumor_threshold_high,
                                                              gen_grid_patches, random_patch_ratio, patches_dir,
                                                              sample_num_tumor))

                # pool.apply(gen_patches_segmentation, (filename_wsi, wsi_level, patch_h, patch_w,
                #                                     file_tumor_mask, tumor_mask_level, tumor_mask_threshold,
                #                                     gen_grid_patches, random_patch_ratio, patches_dir,
                #                                     sample_num_tumor, True))

    pool.close()
    pool.join()

    print('OK')




