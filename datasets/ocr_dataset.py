import json
import os
from typing import Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

from torch_vis.transforms import Transform


class OCRDataset(Dataset):

    def __init__(self, path: str, transforms: Transform):
        self._transforms = transforms
        self._path = path

        self._images_path = os.path.join(path, "img")
        self._annotations_path = os.path.join(path, "ann")

        self._images = list(sorted(os.listdir(self._images_path)))
        self._annotations = list(sorted(os.listdir(self._annotations_path)))

    def __getitem__(self, idx: int) -> Tuple[np.array, str]:
        img_path = os.path.join(self._images_path, self._images[idx])
        annotation_path = os.path.join(self._annotations_path, self._annotations[idx])
        image = cv2.imread(img_path)

        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        label = annotation['description']

        if self._transforms is not None:
            image, label = self._transforms(image, label)

        return image, label

    def __len__(self) -> int:
        return len(self._images)
