import json
from os.path import join

import cv2
import numpy as np


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
        image = cv2.imread(join(image_dir, image_name))
        mask = create_mask(image.shape[:-1], image_data['regions'])
        image_name = ''.join(image_name.split('.')[:-1]) + '.png'
        cv2.imwrite(join(out_image_dir, image_name), image)
        cv2.imwrite(join(mask_dir, image_name), mask)


if __name__ == '__main__':
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
