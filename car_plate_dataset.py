import os
import numpy as np
import torch
from PIL import Image


class CarPlateDataset:

    def __init__(self, path, transforms):
        self._transforms = transforms
        self._path = path

        self._images_path = os.path.join(path, "images")
        self._masks_path = os.path.join(path, "masks")

        self._images = list(sorted(os.listdir(self._images_path)))
        self._masks = list(sorted(os.listdir(self._masks_path)))

    def __getitem__(self, idx):

        img_path = os.path.join(self._images_path, self._images[idx])
        mask_path = os.path.join(self._masks_path, self._masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            x_min = np.min(pos[1])
            x_max = np.max(pos[1])
            y_min = np.min(pos[0])
            y_max = np.max(pos[0])
            boxes.append([x_min, y_min, x_max, y_max])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target

    def __len__(self):
        return len(self._images)

