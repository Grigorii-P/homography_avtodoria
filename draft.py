import itertools
import json
import os
import pickle
from itertools import combinations_with_replacement as cwr
from os.path import join
from time import time
import matplotlib.pyplot as plt
import numpy as np
from annoy import AnnoyIndex
import cv2 as cv
import cv2
from homography import Homography, print_
from homography_data import pts_src_, pts_real_


# with open('../first_video/plates_ever_met_total.p', 'rb') as f:
#     d = pickle.load(f)

# print()


# path_to_video_and_timestamp = '/home/grigorii/Desktop/momentum_speed/first_video'
# video = 'homo_video'

# total_frames = cap.get(7)


# rang = range(1,100,3)
# cap = cv.VideoCapture(join(path_to_video_and_timestamp, video))
# for i in rang:
#     cap.set(cv.CAP_PROP_POS_FRAMES, i)
#     ret, img = cap.read()
#     cv.imshow('frame', img)
#     print(cap.get(1))
#     if cv.waitKey(10) & 0xFF == ord('q'):
#         break

# img = cv.imread('/home/grigorii/Desktop/momentum_speed/video_cruise_control/second_experiment/results/10.mp4/0.jpg')
# img = cv.drawMarker(img, (1000, 1000), (34,139,34), markerType=cv.MARKER_TILTED_CROSS, markerSize=100, thickness=5, line_type=cv.LINE_AA)
# img = cv.resize(img, (1000, 1000))
# cv.imshow('img', img)
# cv.waitKey(0)


A = [105, 1277]
B = [1883, 889]
C = [2914, 661]
D = [3875, 1112]
E = [3621, 2240]
F = [480, 2632]

AB = 10.473
BE = 16.88
BC = 11.940
CD = 11.120
DE = 16.983
EF = 7.266
FA = 11.472
dists = [AB, BC, CD, DE, EF, FA]

hom = Homography(np.array(pts_src_), np.array(pts_real_))

K = [2404, 1294]
L = [2877, 1657]
M = [3440, 904]
# print(hom.get_point_transform_2(hom.h, B, D))

print(cv2.)

# print(hom.get_point_transform_2(hom.h, B, C))
# print(hom.get_point_transform_2(hom.h, C, D))
# print(hom.get_point_transform_2(hom.h, D, E))
# print(hom.get_point_transform_2(hom.h, B, E))


# homs = hom.homs
# errs = []
# for h in homs:
#     err = 0
#     err += AB - hom.get_point_transform_2(h[0], A, B)
#     err += BE - hom.get_point_transform_2(h[0], B, E)
#     err += BC - hom.get_point_transform_2(h[0], B, C)
#     err += CD - hom.get_point_transform_2(h[0], C, D)
#     err += DE - hom.get_point_transform_2(h[0], D, E)
#     err += EF - hom.get_point_transform_2(h[0], E, F)
#     err += FA - hom.get_point_transform_2(h[0], F, A)
#     errs.append([err, h[1], h[2]])
    # print(AB - hom.get_point_transform_2(h, A, B))
    # print(BC - hom.get_point_transform_2(h, B, C))
    # print(CD - hom.get_point_transform_2(h, C, D))
    # print(DE - hom.get_point_transform_2(h, D, E))
    # print(EF - hom.get_point_transform_2(h, E, F))
    # print(FA - hom.get_point_transform_2(h, F, A))
    # print()
# min = 100
# for err in errs:
#     if err[0] < min:
#         min = err[0]
#         item = err
# print('the best one is {} {} {}'.format(err[0], err[1], err[2]))


# im_src = cv.imread('../video_cruise_control/second_experiment/data/Image__2018-10-24__10-16-57.bmp', 0)
# im_out = cv2.warpPerspective(im_src, hom.h, (im_src.shape[0],im_src.shape[1]))
# im_out = cv.resize(im_out, (2000, 1500))
# cv.imshow('im_out', im_out)
# cv.waitKey(0)