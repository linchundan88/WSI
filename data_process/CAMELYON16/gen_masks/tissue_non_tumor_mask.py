
import os
import numpy as np
import tifffile

if __name__ == '__main__':
    level = 4
    tissue_mask_dir = f'F:\\CAMELYON16\\training\\tissue_mask\\{str(level)}\\tumor\\'
    tumor_mask_dir = f'F:\\CAMELYON16\\training\\tumor_mask\\{str(level)}\\'
    tumor_normal_mask_dir = f'F:\\CAMELYON16\\training\\tumor_normal_mask\\{str(level)}\\'

    for dir_path, subpaths, files in os.walk(tissue_mask_dir, False):
        for f in files:
            file_tissue_mask = os.path.join(dir_path, f)
            _, full_filename = os.path.split(file_tissue_mask)
            filename, file_extension = os.path.splitext(full_filename)
            if file_extension.upper() not in ['.NPY']:
                continue

            print(f'operating {file_tissue_mask}...')

            tissue_mask = np.load(file_tissue_mask)
            file_tumor_mask = file_tissue_mask.replace(tissue_mask_dir, tumor_mask_dir)
            tumor_mask = np.load(file_tumor_mask)

            tumor_normal_mask = tissue_mask & (~ tumor_mask)

            file_non_tumor_mask_npy = file_tissue_mask.replace(tissue_mask_dir, tumor_normal_mask_dir)
            print(f'writing {file_non_tumor_mask_npy}...')
            os.makedirs(os.path.dirname(file_non_tumor_mask_npy), exist_ok=True)
            np.save(file_non_tumor_mask_npy, tumor_normal_mask)
            print(f'writing {file_non_tumor_mask_npy} complete.')

            img_tumor_normal_mask = np.array(tumor_normal_mask).astype(np.uint8)
            img_tumor_normal_mask *= 255
            file_non_tumor_mask_npy = file_tissue_mask.replace(tissue_mask_dir, tumor_normal_mask_dir).replace('.npy', '.tif')
            print(f'writing {file_non_tumor_mask_npy}...')
            tifffile.imwrite(file_non_tumor_mask_npy, img_tumor_normal_mask, compress=9)
            print(f'writing {file_non_tumor_mask_npy} complete.')

    print('OK')



