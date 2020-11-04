'''https://github.com/wisepaip/paip2020'''

import glob
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import openslide
import tifffile

wsi_load_dir = '/home/stu/data_share/PAIP2020/wsi/'
xml_load_dir = '/home/stu/data_share/PAIP2020/xml/'
wsi_fns = sorted(glob.glob(wsi_load_dir + '*.svs'))
xml_fns = sorted(glob.glob(xml_load_dir + '*.xml'))
save_dir = '/home/stu/data_share/PAIP2020/mask/'
os.makedirs(save_dir, exist_ok=True)

div = 16  ## Level0 scale to Level2 scale
assert len(wsi_fns) == len(xml_fns) == 47  ## the number of training_data WSI pool

'''
Annotations (root)
> Annotation (get 'Id' -> 1: tumor area)
 > Regions
  > Region (get 'NegativeROA' -> 0: positive area // 1: inner negative area)
   > Vertices
    > Vertex (get 'X', 'Y')
'''


def xml2mask(xml_fn, shape):
    # print('reconstructing sparse xml to contours of div={}..'.format(div))
    ret = dict()

    # Annotations >>
    e = ET.parse(xml_fn).getroot()
    e = e.findall('Annotation')
    assert (len(e) == 1), len(e)

    ann = e[0]
    board_pos = np.zeros(shape[:2], dtype=np.uint8)
    board_neg = np.zeros(shape[:2], dtype=np.uint8)
    id_num = int(ann.get('Id'))
    assert (id_num == 1)  # or id_num == 2)
    rs = ann.findall('Regions/Region')
    assert (len(rs) > 0)
    plistlist = list()
    nlistlist = list()
    print('rs:', len(rs))
    for i, r in enumerate(rs):
        ylist = list()
        xlist = list()
        plist, nlist = list(), list()
        negative_flag = int(r.get('NegativeROA'))
        assert negative_flag == 0 or negative_flag == 1
        negative_flag = bool(negative_flag)
        vs = r.findall('Vertices/Vertex')
        vs.append(vs[0])  # last dot should be linked to the first dot
        for v in vs:
            y, x = int(v.get('Y').split('.')[0]), int(v.get('X').split('.')[0])
            if div is not None:
                y //= div
                x //= div
            if y >= shape[0]:
                y = shape[0] - 1
            elif y < 0:
                y = 0
            if x >= shape[1]:
                x = shape[1] - 1
            elif x < 0:
                x = 0
            ylist.append(y)
            xlist.append(x)
            if negative_flag:
                nlist.append((x, y))
            else:
                plist.append((x, y))
        if plist:
            plistlist.append(plist)
        else:
            nlistlist.append(nlist)

    for plist in plistlist:
        board_pos = cv2.drawContours(board_pos, [np.array(plist, dtype=np.int32)], -1, [255, 0, 0], -1)
    for nlist in nlistlist:
        board_neg = cv2.drawContours(board_neg, [np.array(nlist, dtype=np.int32)], -1, [255, 0, 0], -1)
    ret[id_num] = (board_pos > 0) * (board_neg == 0)


    return ret


def save_mask(wsi_id, xml_fn, shape):
    save_fn = os.path.join(save_dir + '{}_l2_annotation_tumor.tif'.format(wsi_id))
    ret = xml2mask(xml_fn, shape)
    tifffile.imsave(save_fn, (ret[1] > 0).astype(np.uint8), compress=9)

def load_svs_shape(fn, level=2):
    imgh = openslide.OpenSlide(fn)
    return [imgh.level_dimensions[level][1], imgh.level_dimensions[level][0]]


if __name__ == '__main__':

    for i, (wsi_fn, xml_fn) in enumerate(zip(wsi_fns, xml_fns)):
        _, filename = os.path.split(wsi_fn)
        wsi_id, _ = os.path.splitext(filename)
        _, filename = os.path.split(xml_fn)
        xml_id, _ = os.path.splitext(filename)

        assert wsi_id == xml_id
        assert os.path.isfile(wsi_fn) and os.path.isfile(xml_fn)
        print(i + 1, '/', len(wsi_fns), ':', wsi_id)
        shape = load_svs_shape(wsi_fn, level=2)
        save_mask(wsi_id, xml_fn, shape)



# https://paip2019.grand-challenge.org/Dataset/




