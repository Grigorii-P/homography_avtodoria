import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
# from homography import *
# from homography_data import *


path_to_video = '/home/grigorii/Desktop/momentum_speed/homo_video'


cap = cv2.VideoCapture(path_to_video)


num = 30
cap.set(1, 80)
for i in range(num):
    ret, img = cap.read()
    # img = cv2.drawMarker(img, (250,250), (0,0,255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=3, line_type=cv2.LINE_AA)
    cv2.imwrite('../temp/' + str(i) + '.jpg', img)
