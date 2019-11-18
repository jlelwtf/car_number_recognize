import json
import os
from os.path import join

import cv2
import numpy as np


stop_list = [
    '236086200orig',
    '236273765orig',
    '239940611orig',
    '242294187orig',
    '249534768orig',
    '250018121orig',
    '250019884orig',
    '250020320orig',
    '250026658orig',
    '250029126orig',
    '236086200orig',
    '236273765orig',
    '239940611orig',
    '242294187orig',
    '208987567orig',
    '248429499orig',
    '249283330orig',
    '250022241orig',
    '250034165orig',
]


def create_mask(shape, regions):
    mask = np.zeros(shape)
    for idx, region in enumerate(regions):
        x_points = region['shape_attributes']['all_points_x']
        y_points = region['shape_attributes']['all_points_y']
        points = np.array([[(x, y) for x, y in zip(x_points, y_points)]])
        pxval = (idx + 1) * 10
        cv2.fillPoly(mask, points, pxval)
    return mask


def generate_dataset(label_data_path, image_dir, out_image_dir, mask_dir):
    with open(label_data_path, 'r') as f:
        data = json.load(f)
    meta = data['_via_img_metadata']
    for _, image_data in meta.items():
        image_name = image_data['filename']
        base_image_name = ''.join(image_name.split('.')[:-1])
        if base_image_name not in stop_list:
            image = cv2.imread(join(image_dir, image_name))
            mask = create_mask(image.shape[:-1], image_data['regions'])
            cv2.imwrite(join(out_image_dir, base_image_name + '.png'), image)
            cv2.imwrite(join(mask_dir, base_image_name + '.png'), mask)


if __name__ == '__main__':

    os.makedirs('data/masked_number_plates/train/images')
    os.makedirs('data/masked_number_plates/train/masks')
    os.makedirs('data/masked_number_plates/val/images')
    os.makedirs('data/masked_number_plates/val/masks')

    generate_dataset(
        'data/number_plates/train/via_region_data.json',
        'data/number_plates/train',
        'data/masked_number_plates/train/images',
        'data/masked_number_plates/train/masks'
    )

    generate_dataset(
        'data/number_plates/val/via_region_data.json',
        'data/number_plates/val',
        'data/masked_number_plates/val/images',
        'data/masked_number_plates/val/masks'
    )
