import random
from abc import abstractmethod

import torch
from typing import Tuple, List, Union

import cv2
import imutils
import numpy as np
from torchvision.transforms.functional import to_tensor


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Transform:
    @abstractmethod
    def __call__(self, image: np.array, target: Union[str, np.array, dict]):
        raise NotImplementedError


class Compose(Transform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(Transform):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToRCNNTensor(Transform):
    def __call__(self, image, target):
        image = to_tensor(image)
        return image, target


class ToOCRTensor(Transform):
    def __init__(
            self,
            symbol_list: List[str],
            num_time_steps: int,
            max_sequence_len: int,
            space_symbol_idx: int
    ):
        self._space_symbol_idx = space_symbol_idx
        self._max_sequence_len = max_sequence_len
        self._symbol_list = symbol_list
        self._num_classes = len(symbol_list)
        self._num_time_steps = num_time_steps

    def _create_target(self, label: str) -> np.array:
        target = []

        for symbol in label:
            target.append(self._symbol_list.index(symbol) + 1)

        for _ in range(self._max_sequence_len - len(target)):
            target.append(0)

        return np.array(target)

    def __call__(self, image: np.array, label: str) -> Tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(image).float()
        target = self._create_target(label)
        target = torch.from_numpy(target).int()
        return image, target


class TransformRuLabel(Transform):
    def __init__(self, sep_symbol: str, space_symbol: str):
        self._space_symbol = space_symbol
        self._sep_symbol = sep_symbol

    def _transform_label(self, label: str) -> str:
        number = self._sep_symbol.join(label[:6])
        region = self._sep_symbol.join(label[6:])
        # number = ''.join(label[:6])
        # region = ''.join(label[6:])
        return number + self._space_symbol + region

    def __call__(self, image: np.array, label: str) -> Tuple[np.array, str]:
        label = self._transform_label(label)
        return image, label


class TransformImageForOCR(Transform):
    def __init__(self, image_height: int, image_width: int):
        self._image_height = image_height
        self._image_width = image_width

    def _transform_image(self, image: np.array) -> np.array:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = imutils.rotate_bound(image, 90)
        image = cv2.resize(image, (self._image_width, self._image_height))
        image = np.expand_dims(image, 0)
        return image

    def __call__(self, image: np.array, label: str) -> Tuple[np.array, str]:
        image = self._transform_image(image)
        return image, label




