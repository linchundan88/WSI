import os
import random
import pandas as pd
import csv
import sklearn

def write_csv_based_on_dir(filename_csv, img_dir, mask_dir,
       list_file_ext=['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']):

    if os.path.exists(filename_csv):
        os.remove(filename_csv)
    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)

    if not img_dir.endswith('/'):
        img_dir += '/'
    if not mask_dir.endswith('/'):
        mask_dir += '/'

    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'masks'])

        for dir_path, subpaths, files in os.walk(img_dir, False):
            for f in files:
                file_image = os.path.join(dir_path, f)
                (filedir, filename) = os.path.split(file_image)
                (file_basename, extension) = os.path.splitext(filename)
                if extension.upper() not in list_file_ext:
                    print('file ext name:', f)
                    continue

                file_mask = file_image.replace(img_dir, mask_dir)
                if os.path.exists(file_mask):
                    print(file_mask)
                    csv_writer.writerow([file_image, file_mask])


