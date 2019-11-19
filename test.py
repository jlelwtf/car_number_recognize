from time import time

import cv2

from models.mask_rcnn import MaskRCNN
from number_plate_detector import NumberPlateDetector

if __name__ == '__main__':
    rcnn = MaskRCNN(device='cpu')
    rcnn.load('models/data/mask-rcnn.pth')
    detector = NumberPlateDetector(rcnn)
    # image = cv2.imread('datasets/masked_number_plates/val/images/250023138orig.png')
    image = cv2.imread('test2.jpg')
    t = time()
    res = detector.detect_number(image)
    print(time() - t)
    t = time()
    res = detector.detect_number(image)
    print(time() - t)

    if res is not None:
        cv2.imshow('asd', res)
        cv2.waitKey(0)
