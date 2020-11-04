
import os
import xml.etree.cElementTree as ET
import cv2
import tifffile
import numpy as np
import openslide
import gc


if __name__ == '__main__':
    xml_dir = 'F:\\CAMELYON16\\training\\lesion_annotations\\'
    wsi_dir = 'F:\\CAMELYON16\\training\\wsi\\tumor\\'
    mask_dir = 'F:\\CAMELYON16\\training\\tumor_mask\\'

    level = 5

    for dir_path, subpaths, files in os.walk(xml_dir, False):
        for f in files:
            file_xml = os.path.join(dir_path, f)
            _, full_filename = os.path.split(file_xml)
            filename, file_extension = os.path.splitext(full_filename)

            print(f'operate {file_xml}...')
            filename_wsi = file_xml.replace(xml_dir, wsi_dir).replace('.xml', '.tif')
            slide = openslide.open_slide(filename_wsi)

            (width, height) = slide.level_dimensions[level]
            img_mask = np.zeros((height, width), dtype=np.uint8)
            factor = slide.level_downsamples[level]

            tree = ET.parse(file_xml)
            root = tree.getroot()  # root[0]  #Annotations
            e_annotations = root[0].findall('Annotation')

            for e_annotation in e_annotations:
                if e_annotation.attrib['PartOfGroup'] in ['_0', '_1']:
                    # e_coordinates = e_annotation[0] OK, each Annotation only contain one <Coordinates>
                    for e_coordinates in e_annotation:
                        plist_positive = list()
                        for e_coordinate in e_coordinates:
                            x = round(float(e_coordinate.attrib['X']))
                            y = round(float(e_coordinate.attrib['Y']))
                            # order = e_coordinate.attrib['Order']
                            plist_positive.append((int(x/factor), int(y/factor)))

                        cv2.fillPoly(img_mask, [np.array(plist_positive)], (255))

            for e_annotation in e_annotations:
                if e_annotation.attrib['PartOfGroup'] in ['_2']:
                    for e_coordinates in e_annotation:
                        plist_negative = list()
                        for e_coordinate in e_coordinates:
                            x = round(float(e_coordinate.attrib['X']))
                            y = round(float(e_coordinate.attrib['Y']))
                            # order = e_coordinate.attrib['Order']
                            plist_negative.append((int(x / factor), int(y / factor)))

                        cv2.fillPoly(img_mask, [np.array(plist_negative)], (0))

            file_tumor_mask_npy = os.path.join(mask_dir, str(level), filename + '.npy')
            tumor_mask = img_mask > 127
            os.makedirs(os.path.dirname(file_tumor_mask_npy), exist_ok=True)
            print(f'start writing {file_tumor_mask_npy}.')
            np.save(file_tumor_mask_npy, tumor_mask)
            print(f'writing {file_tumor_mask_npy} complete.')

            file_tumor_mask = os.path.join(mask_dir, str(level), filename + '.tif')
            print(f'start writing {file_tumor_mask}.')
            tifffile.imwrite(file_tumor_mask, img_mask, compress=9)
            print(f'writing {file_tumor_mask} complete.')

            slide.close()
            del img_mask
            gc.collect()

    print('OK')

