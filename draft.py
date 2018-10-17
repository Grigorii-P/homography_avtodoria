import itertools
import json
import os
from itertools import combinations_with_replacement as cwr
from os.path import join
from time import time
import matplotlib.pyplot as plt
import numpy as np
from annoy import AnnoyIndex
import cv2 as cv


# path_to_video_and_timestamp = '/home/grigorii/Desktop/momentum_speed/video_cruise_control'
# video = 'regid_1538565891498_ffv1_45'

# cap = cv.VideoCapture(join(path_to_video_and_timestamp, video))
# c = 0
# while True:
#     ret, img = cap.read()
#     c += 1
#     if ret is False:
#         break
#     print(c)


path_to_imgs = '../repers/annoy_test'

img_size = (450, 90)
templates = os.listdir(path_to_imgs)
templates = [x for x in templates if x.endswith('.jpg') and '@' in x]

# for i, temp in enumerate(templates):
#     img = cv.imread(join(path_to_imgs, temp), 0) #'A283CO716@_76.jpg'
#     x, y = img.shape[1] // 2, img.shape[0] // 2
#     img = cv.drawMarker(img, (x, y), (255,0,0), markerType=cv.MARKER_TILTED_CROSS, markerSize=10, thickness=1, line_type=cv.LINE_AA)
#     img = cv.resize(img, img_size)
#     # cv.imwrite(join(path_to_imgs, 'first.jpg'), img)
#     cv.imshow('asd', img)
#     cv.waitKey(0)


for i, temp in enumerate(templates):
    img = cv.imread(join(path_to_imgs, temp), 0)
    x, y = img.shape[1] // 2, img.shape[0] // 2
    img = cv.drawMarker(img, (x, y), (255,0,0), markerType=cv.MARKER_TILTED_CROSS, markerSize=10, thickness=1, line_type=cv.LINE_AA)
    img = cv.resize(img, img_size)
    cv.imwrite(join(path_to_imgs, str(i) + str(i) + '.jpg'), img)
