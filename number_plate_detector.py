import imutils
import torch
from typing import Optional

import cv2
import numpy as np
from torchvision.transforms import functional as F
from models.mask_rcnn import MaskRCNN


class NumberPlateDetector:

    def __init__(self, mask_rcnn_model: MaskRCNN):
        self._mask_rcnn_model = mask_rcnn_model

    @staticmethod
    def _get_rect(mask: np.array) -> Optional[np.array]:
        if cv2.__version__[0] == "4":
            cnts, _ = cv2.findContours(mask, 1, 2)
        else:
            cnts = cv2.findContours(mask, 1, 2)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        epsilon = 0.05 * cv2.arcLength(cnts[0], True)
        approx = cv2.approxPolyDP(cnts[0], epsilon, True)
        if len(approx) != 4:
            return None
        return approx.reshape((4, 2)).astype(np.float32)


    @staticmethod
    def _convert_mask(mask: torch.Tensor) -> np.array:
        np_mask = mask.cpu().detach().numpy()[0]
        np_mask[np_mask > 0.6] = 1
        np_mask[np_mask <= 0.6] = 0
        np_mask = np_mask.astype(np.uint8)
        np_mask = (np_mask + 1) % 2
        return np_mask

    @staticmethod
    def _order_points(pts: np.array) -> np.array:
        """
        from:
        https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
        """
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def _four_point_transform(self, image: np.array, pts: np.array) -> np.array:
        """
        from:
        https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
        """
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def detect_number(self, image: np.array) -> Optional[np.array]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(image).to(self._mask_rcnn_model.device)
        result = self._mask_rcnn_model.predict([img_tensor])
        if result:
            detect = result[0]
            mask = self._convert_mask(detect['masks'][0])
            rect = self._get_rect(mask)
            if rect is not None:
                return self._four_point_transform(image, rect)
        return None




