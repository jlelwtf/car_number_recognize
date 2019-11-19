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
    for idx, (_, image_data) in enumerate(meta.items()):
        print(f'{idx} from {len(meta)}', end='\r')

        image_name = image_data['filename']
        base_image_name = ''.join(image_name.split('.')[:-1])
        if base_image_name not in stop_list:
            image = cv2.imread(join(image_dir, image_name))
            mask = create_mask(image.shape[:-1], image_data['regions'])
            cv2.imwrite(join(out_image_dir, base_image_name + '.png'), image)
            cv2.imwrite(join(mask_dir, base_image_name + '.png'), mask)
    print()


if __name__ == '__main__':
    os.system(
        'wget '
        'https://nomeroff.net.ua/datasets/autoriaNumberplateDataset-2019-03-06.zip '
        '-P datasets'
    )

    os.system(
        'unzip '
        'datasets/autoriaNumberplateDataset-2019-03-06.zip '
        '-d datasets'
    )

    os.system(
        'mv '
        'datasets/autoriaNumberplateDataset-2019-03-06 '
        'datasets/number_plates'
    )

    os.system(
        'rm '
        'datasets/autoriaNumberplateDataset-2019-03-06.zip '
    )

    os.makedirs('data/masked_number_plates/train/images')
    os.makedirs('data/masked_number_plates/train/masks')
    os.makedirs('data/masked_number_plates/val/images')
    os.makedirs('data/masked_number_plates/val/masks')

    print('Generating masks for train dataset...')
    generate_dataset(
        'datasets/number_plates/train/via_region_data.json',
        'datasets/number_plates/train',
        'datasets/masked_number_plates/train/images',
        'datasets/masked_number_plates/train/masks'
    )

    print('Generating masks for val dataset...')
    generate_dataset(
        'datasets/number_plates/val/via_region_data.json',
        'datasets/number_plates/val',
        'datasets/masked_number_plates/val/images',
        'datasets/masked_number_plates/val/masks'
    )

    os.system('rm -rf datasets/number_plates')
