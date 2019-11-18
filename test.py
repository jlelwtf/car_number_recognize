from PIL import Image
import numpy as np

from car_plate_dataset import CarPlateDataset
from main import get_transform

if __name__ == '__main__':
    # mask = Image.open('data/masked_number_plates/train/masks/c867f1fe1e7f46c56df3b1cf649e5b91.png')
    # mask = np.array(mask)
    # obj_ids = np.unique(mask)
    # obj_ids = obj_ids[1:]
    #
    # masks = mask == obj_ids[:, None, None]
    #
    # num_objs = len(obj_ids)
    # boxes = []
    # for i in range(num_objs):
    #     pos = np.where(masks[i])
    #     print(pos)
    #     x_min = np.min(pos[1])
    #     x_max = np.max(pos[1])
    #     y_min = np.min(pos[0])
    #     y_max = np.max(pos[0])
    #     boxes.append([x_min, y_min, x_max, y_max])
    dataset = CarPlateDataset(
        'data/masked_number_plates/train', get_transform(train=True)
    )
    dataset_test = CarPlateDataset(
        'data/masked_number_plates/val', get_transform(train=False)
    )
    # val = dataset[181]
    #
    for idx in range(len(dataset)):
        try:
            val = dataset[idx]
        except:
            print(dataset.get_image_name(idx))
