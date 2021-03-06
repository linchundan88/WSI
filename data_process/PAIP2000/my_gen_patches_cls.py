
import os
import multiprocessing
from libs.dataPreprocess.my_patches import gen_patches_classification, gen_patches_segmentation


if __name__ == '__main__':

    wsi_level = 0   #PAIP2000 slide.level_downsamples(0:1, 1:4, 2:16, 3:32)
    tissue_level = 2
    tumor_level = 2

    wsi_dir = '/home/stu/data_share/PAIP2020/wsi/'
    tissue_mask_dir = f'/home/stu/data_share/PAIP2020/tissue_mask/'
    tumor_mask_dir = f'/home/stu/data_share/PAIP2020//tumor_mask/'

    # patch_h, patch_w = (320, 320)
    patch_h, patch_w = (299, 299)

    patches_dir_base = f'/home/stu/data_share/PAIP2020/patches/classification/wsi_level{str(wsi_level)}_{str(patch_h)}_{str(patch_w)}/'

    # sample_num_normal, sample_num_tumor = (3000, 1000)
    sample_num_normal, sample_num_tumor = (None, None)

    tissue_threshold = 15
    tumor_threshold_lower = 0.1
    tumor_threshold_high = 2

    multi_processes_num = 8
    pool = multiprocessing.Pool(processes=multi_processes_num)

    for dir_path, subpaths, files in os.walk(wsi_dir, False):
        for f in files:
            filename_wsi = os.path.join(dir_path, f)
            _, filename = os.path.split(filename_wsi)
            filename_base, file_extension = os.path.splitext(filename)
            if file_extension.upper() not in ['.SVS']:
                continue

            file_tissue = filename_wsi.replace(wsi_dir, tissue_mask_dir).replace('.svs', '_l2_annotation_tissue.tif')
            assert os.path.exists(file_tissue), f'{file_tissue} not exist.'
            file_tumor = filename_wsi.replace(wsi_dir, tumor_mask_dir).replace('.svs', '_l2_annotation_tumor.tif')
            assert os.path.exists(file_tissue), f'{file_tumor} not exist.'

            patches_dir = os.path.join(patches_dir_base, '0', filename_base)
            gen_grid_patches = True
            random_patch_ratio = None
            pool.apply_async(gen_patches_classification, (filename_wsi, wsi_level, patch_h, patch_w,
                                                          patches_dir, gen_grid_patches, random_patch_ratio, sample_num_normal,
                                                          file_tissue, tissue_level, tissue_threshold,
                                                          file_tumor, tumor_level, tumor_threshold_lower, 'exclude'))

            patches_dir = os.path.join(patches_dir_base, '1', filename_base)
            gen_grid_patches = True
            random_patch_ratio = None
            pool.apply_async(gen_patches_classification, (filename_wsi, wsi_level, patch_h, patch_w,
                                                          patches_dir, gen_grid_patches, random_patch_ratio, sample_num_tumor,
                                                          file_tumor, tumor_level, tumor_threshold_high))


    pool.close()
    pool.join()

    print('OK')



