from libs.dataPreprocess.my_patches import gen_validate_patches, gen_test_patches

if __name__ == '__main__':

    filename_wsi = '/home/stu/data_share/PAIP2020/wsi/training_data_01.svs'
    wsi_level = 0  # the WSI level that extract patches from

    file_tissue = '/home/stu/data_share/PAIP2020/tissue_mask/training_data_01_l2_annotation_tissue.tif'
    tissue_level = 2
    tissue_threshold = 15

    file_tumor = '/home/stu/data_share/PAIP2020/tumor_mask/training_data_01_l2_annotation_tumor.tif'
    tumor_level = 2
    tumor_threshold = 0.1

    patch_shape = (299, 299)
    patches_dir_base = '/home/stu/data_share/PAIP2020/predict/patches_validate'
    filename_csv = '/home/stu/data_share/PAIP2020/predict/patches_validate.csv'

    gen_validate_patches(filename_wsi, wsi_level,
                         file_tissue, tissue_level, tissue_threshold,
                         file_tumor, tumor_level, tumor_threshold,
                         patch_shape, patches_dir_base, filename_csv)


    patches_dir_base = '/home/stu/data_share/PAIP2020/predict/patches_test'
    filename_csv = '/home/stu/data_share/PAIP2020/predict/patches_test.csv'

    gen_test_patches(filename_wsi, wsi_level,
                     file_tissue, tissue_level, tissue_threshold,
                     patch_shape, patches_dir_base, filename_csv)

    print('OK')





