import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from homography import *
from homography_data import *

# path_to_video = '../video_cruise_control/regid_1538565891498_ffv1_45'

# cap = cv2.VideoCapture(path_to_video)
# ret, img = cap.read()

# cv2.imwrite('temp.jpg', img)


hom = Homography(pts_src_, pts_real_)

dist = hom.get_point_transform([1136, 184], [942, 363])
print()