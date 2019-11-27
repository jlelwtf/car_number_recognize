import os
from time import time

import cv2

from models.mask_rcnn import MaskRCNN
from number_plate_detector import NumberPlateDetector

if __name__ == '__main__':
    rcnn = MaskRCNN(device='cpu')
    rcnn.load('models/data/mask-rcnn.pth')
    detector = NumberPlateDetector(rcnn)
    # image = cv2.imread('datasets/masked_number_plates/val/images/250023138orig.png')
    for image_name in os.listdir('test_set'):
        print(image_name)
        image = cv2.imread('test_set/' + image_name)
        images, rects = detector.detect_number(image)

        for image in images:
            # cv2.line(image, (rect[0, 0], rect[0, 1]), (rect[1, 0], rect[1, 1]), (0, 255, 0), 2)
            # cv2.line(image, (rect[1, 0], rect[1, 1]), (rect[2, 0], rect[2, 1]), (0, 255, 0), 2)
            # cv2.line(image, (rect[2, 0], rect[2, 1]), (rect[3, 0], rect[3, 1]), (0, 255, 0), 2)
            # cv2.line(image, (rect[3, 0], rect[3, 1]), (rect[0, 0], rect[0, 1]), (0, 255, 0), 2)
            # cv2.polylines(image, cnts, True, (0, 255, 0), 2)
            cv2.imshow('as', image)
            cv2.waitKey(0)
