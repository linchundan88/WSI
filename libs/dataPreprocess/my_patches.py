import csv
import gc
import os
import random
import uuid
import cv2
import numpy as np
import openslide
import tifffile
from openslide.deepzoom import DeepZoomGenerator

def extract_bbox(patch_h_level0, patch_w_level0, gen_grid_patches, random_patch_ratio,
                 file_tissue, factor_tissue, tissue_threshold,
                 file_tumor=None, factor_tumor=None, tumor_threshold=None, relation='exclude'):

    image_mask = tifffile.imread(file_tissue)
    img_h, img_w = image_mask.shape[0:2]

    if file_tumor is not None:
        image_mask_tumor = tifffile.imread(file_tumor)

    # patch size(level 0) project to the tumor image
    patch_h = int(patch_h_level0 / factor_tissue)
    patch_w = int(patch_w_level0 / factor_tissue)

    list_bbox = list()
    if gen_grid_patches:
        for num_y in range(int(img_h / patch_h)):
            for num_x in range(int(img_w / patch_w)):
                list_bbox.append((num_y * patch_h, (num_y+1) * patch_h, num_x * patch_w, (num_x+1) * patch_w))
    if random_patch_ratio is not None:
        patch_random_num = int(len(list_bbox) * random_patch_ratio)
        for _ in range(patch_random_num):
            y = random.randint(0, img_h - patch_h)
            x = random.randint(0, img_w - patch_w)
            list_bbox.append((y, y + patch_h, x, x + patch_w))

    list_bbox_results = list()

    for (y1, y2, x1, x2) in list_bbox:
        if np.mean(image_mask[y1:y2, x1:x2]) >= tissue_threshold:
            if file_tumor is None:
                y1 = int(y1 * factor_tissue)
                y2 = int(y2 * factor_tissue)
                x1 = int(x1 * factor_tissue)
                x2 = int(x2 * factor_tissue)
                list_bbox_results.append((y1, y2, x1, x2))
            else:
                mean_tumor_value = np.mean(image_mask_tumor[int(y1 * factor_tissue / factor_tumor):int(y2 * factor_tissue / factor_tumor),
                                int(x1 * factor_tissue / factor_tumor):int(x2 * factor_tissue / factor_tumor)])

                assert relation in ['exclude', 'include'], 'relation type error'
                if (relation == 'exclude' and mean_tumor_value <= tumor_threshold)\
                            or (relation == 'include' and mean_tumor_value >= tumor_threshold):
                    y1 = int(y1 * factor_tissue)
                    y2 = int(y2 * factor_tissue)
                    x1 = int(x1 * factor_tissue)
                    x2 = int(x2 * factor_tissue)
                    list_bbox_results.append((y1, y2, x1, x2))


    return list_bbox_results



def gen_patches_classification(file_wsi, wsi_level, patch_h, patch_w,
                               patches_dir, gen_grid_patches, random_patch_ratio, sample_num,
                               file_tissue, tissue_level, tissue_threshold,
                               file_tumor=None, tumor_level=None, tumor_threshold=None,
                               relation='exclude'):

    _, filename = os.path.split(file_wsi)
    filename_base, _ = os.path.splitext(filename)

    slide = openslide.OpenSlide(file_wsi)
    #patch size is based on wsi_level, patch size(in this wsi_level) project to level 0
    factor_wsi = slide.level_downsamples[wsi_level]
    patch_h_level0 = patch_h * int(factor_wsi)
    patch_w_level0 = patch_w * int(factor_wsi)

    factor_tissue = slide.level_downsamples[tissue_level]
    if tumor_level is not None:
        factor_tumor = slide.level_downsamples[tumor_level]
    else:
        factor_tumor = None

    list_bbox = extract_bbox(patch_h_level0=patch_h_level0, patch_w_level0=patch_w_level0,
                             gen_grid_patches=gen_grid_patches, random_patch_ratio=random_patch_ratio,
                             file_tissue=file_tissue, factor_tissue=factor_tissue, tissue_threshold=tissue_threshold,
                             file_tumor=file_tumor, factor_tumor=factor_tumor, tumor_threshold=tumor_threshold,
                             relation=relation)

    if sample_num is not None and len(list_bbox) > sample_num:
        list_bbox = random.sample(list_bbox, sample_num)

    for index, (y1, y2, x1, x2) in enumerate(list_bbox):
        # location (tuple) – (x, y) tuple giving the top left pixel in the level 0 reference frame
        img_patch = np.array(slide.read_region((x1, y1), wsi_level, (patch_w, patch_h)).convert('RGB'))

        file_patch = os.path.join(patches_dir, str(index // 100), f'{str(uuid.uuid4())}.jpg')
        os.makedirs(os.path.dirname(file_patch), exist_ok=True)
        print(file_patch)
        cv2.imwrite(file_patch, img_patch)


    slide.close()
    gc.collect()


def gen_patches_segmentation(filename_wsi, wsi_level, patch_h, patch_w,
                             patches_dir_images, patches_dir_masks,
                             gen_grid_patches, random_patch_ratio, sample_num,
                             file_tissue, tissue_level, tissue_threshold,
                             file_tumor, tumor_level
                             ):

    _, filename = os.path.split(filename_wsi)
    filename_base, _ = os.path.splitext(filename)
    slide = openslide.OpenSlide(filename_wsi)
    #patch size is based on wsi_level
    factor_wsi = slide.level_downsamples[wsi_level]
    # patch size(in this wsi_level) project to level 0
    patch_h_level0 = patch_h * int(factor_wsi)
    patch_w_level0 = patch_w * int(factor_wsi)

    factor_tissue = slide.level_downsamples[tissue_level]

    img_tumor = tifffile.imread(file_tumor)
    if tumor_level is not None:
        factor_tumor = slide.level_downsamples[tumor_level]
    else:
        factor_tumor = None


    list_bbox = extract_bbox(patch_h_level0=patch_h_level0, patch_w_level0=patch_w_level0,
                             gen_grid_patches=gen_grid_patches, random_patch_ratio=random_patch_ratio,
                             file_tissue=file_tissue, factor_tissue=factor_tissue, tissue_threshold=tissue_threshold)

    if sample_num is not None and len(list_bbox) > sample_num:
        list_bbox = random.sample(list_bbox, sample_num)

    for index, (y1, y2, x1, x2) in enumerate(list_bbox):
        img_patch = np.array(slide.read_region((x1, y1), wsi_level, (patch_w, patch_h)).convert('RGB'))
        tmp_uuid = str(uuid.uuid4())

        file_patch_image = os.path.join(patches_dir_images, str(index // 100), f'{tmp_uuid}.jpg')
        os.makedirs(os.path.dirname(file_patch_image), exist_ok=True)
        cv2.imwrite(file_patch_image, img_patch)

        mask_tumor = img_tumor[int(y1 / factor_tumor):int(y2 / factor_tumor),
                     int(x1 / factor_tumor):int(x2 / factor_tumor)]
        if mask_tumor.shape[0:2] != (patch_h, patch_w):
            mask_tumor = cv2.resize(mask_tumor, (patch_w, patch_h))

        file_patch_mask = os.path.join(patches_dir_masks, str(index // 100),  f'{tmp_uuid}.jpg')
        os.makedirs(os.path.dirname(file_patch_mask), exist_ok=True)
        cv2.imwrite(file_patch_mask, mask_tumor)

        print(file_patch_image, file_patch_mask)

    slide.close()
    gc.collect()


def gen_test_patches(filename_wsi, wsi_level,
                     file_tissue, tissue_level, tissue_threshold,
                     patch_shape, patches_dir_base, filename_csv):

    patch_h, patch_w = patch_shape
    file_dir, filename = os.path.split(filename_wsi)
    filename_base, file_ext = os.path.splitext(filename)
    patches_dir = os.path.join(patches_dir_base, filename_base)

    slide = openslide.OpenSlide(filename_wsi)

    '''
    tile_size = 384
    data_gen = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0, limit_bounds=False)
    width, height = slide.level_dimensions[0]
    row_num = width // tile_size
    col_num = height // tile_size
    for row in range(row_num):
        for col in range(col_num):
            tile_img1 = data_gen.get_tile(wsi_level, (col, row))  # (col, row)
            # read_region = data_gen.get_tile_coordinates(wsi_level, (col, row))
    '''

    gen_grid_patches = True
    random_patch_ratio = None

    # patch size is based on wsi_level, patch size(in this wsi_level) project to level 0
    factor_wsi = slide.level_downsamples[wsi_level]
    patch_h_level0 = patch_h * int(factor_wsi)
    patch_w_level0 = patch_w * int(factor_wsi)

    factor_mask = slide.level_downsamples[tissue_level]


    list_bbox = extract_bbox(patch_h_level0=patch_h_level0, patch_w_level0=patch_w_level0,
                             gen_grid_patches=gen_grid_patches, random_patch_ratio=random_patch_ratio,
                             file_tissue=file_tissue, factor_tissue=factor_mask, tissue_threshold=tissue_threshold
                             )

    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)
    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'labels', 'y1', 'y2', 'x1', 'x2'])

        for index, (y1, y2, x1, x2) in enumerate(list_bbox):
            img_patch = np.array(slide.read_region((x1, y1), wsi_level, (patch_w, patch_h)).convert('RGB'))
            file_patch = os.path.join(patches_dir, str(index // 100), f'{str(uuid.uuid4())}.jpg')
            os.makedirs(os.path.dirname(file_patch), exist_ok=True)
            print(file_patch)
            cv2.imwrite(file_patch, img_patch)
            csv_writer.writerow([file_patch, -1, y1, y2, x1, x2])


def gen_validate_patches(filename_wsi, wsi_level,
                         file_tissue, tissue_level, tissue_threshold,
                         file_tumor, tumor_level, tumor_threshold,
                         patch_shape, patches_dir_base, filename_csv):

    patch_h, patch_w = patch_shape
    file_dir, filename = os.path.split(filename_wsi)
    filename_base, file_ext = os.path.splitext(filename)
    patches_dir = os.path.join(patches_dir_base, filename_base)
    gen_grid_patches = True
    random_patch_ratio = None

    slide = openslide.OpenSlide(filename_wsi)
    # patch size is based on wsi_level, patch size(in this wsi_level) project to level 0
    factor_wsi = slide.level_downsamples[wsi_level]
    patch_h_level0 = patch_h * int(factor_wsi)
    patch_w_level0 = patch_w * int(factor_wsi)

    factor_mask = slide.level_downsamples[tissue_level]
    factor_tumor = slide.level_downsamples[tumor_level]

    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)

    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'labels', 'y1', 'y2', 'x1', 'x2'])

        # region non tumor patches
        list_bbox = extract_bbox(patch_h_level0=patch_h_level0, patch_w_level0=patch_w_level0,
                                 gen_grid_patches=gen_grid_patches, random_patch_ratio=random_patch_ratio,
                                 file_tissue=file_tissue, factor_tissue=factor_tumor, tissue_threshold=tissue_threshold,
                                 file_tumor=file_tumor, factor_tumor=factor_tumor,
                                 tumor_threshold=tumor_threshold
                                 )

        for index, (y1, y2, x1, x2) in enumerate(list_bbox):
            img_patch = np.array(slide.read_region((x1, y1), wsi_level, (patch_w, patch_h)).convert('RGB'))
            file_patch = os.path.join(patches_dir, '0',  str(index // 100),  f'{str(uuid.uuid4())}.jpg')
            os.makedirs(os.path.dirname(file_patch), exist_ok=True)
            print(file_patch)
            cv2.imwrite(file_patch, img_patch)
            csv_writer.writerow([file_patch, 0, y1, y2, x1, x2])

        # endregion

        #region tumor patches
        list_bbox = extract_bbox(patch_h_level0=patch_h_level0, patch_w_level0=patch_w_level0,
                                 gen_grid_patches=gen_grid_patches, random_patch_ratio=random_patch_ratio,
                                 file_tissue=file_tumor, factor_tissue=factor_tumor, tissue_threshold=tumor_threshold,
                                 )

        for index, (y1, y2, x1, x2) in enumerate(list_bbox):
            img_patch = np.array(slide.read_region((x1, y1), wsi_level, (patch_w, patch_h)).convert('RGB'))
            file_patch = os.path.join(patches_dir, '1', str(index // 100), f'{str(uuid.uuid4())}.jpg')
            os.makedirs(os.path.dirname(file_patch), exist_ok=True)
            print(file_patch)
            cv2.imwrite(file_patch, img_patch)
            csv_writer.writerow([file_patch, 1, y1, y2, x1, x2])

        #endregion