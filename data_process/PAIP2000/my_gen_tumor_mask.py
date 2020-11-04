
import os
import tifffile


mask_dir = '/home/stu/data_share/PAIP2020/mask/'
mask_dir1 = '/home/stu/data_share/PAIP2020/tumor_mask/'
os.makedirs(mask_dir1, exist_ok=True)


for dir_path, subpaths, files in os.walk(mask_dir, False):
    for f in files:
        filename_mask = os.path.join(dir_path, f)
        _, filename = os.path.split(filename_mask)

        img1 = tifffile.imread(filename_mask)
        img2 = img1 * 255

        filename_mask_save = filename_mask.replace(mask_dir, mask_dir1)
        print(filename_mask_save)
        tifffile.imsave(filename_mask_save, img2, compress=9)

print('OK')