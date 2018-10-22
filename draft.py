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


# with open('../first_video/plates_ever_met_total.p', 'rb') as f:
#     d = pickle.load(f)

# print()


path_to_video_and_timestamp = '/home/grigorii/Desktop/momentum_speed/first_video'
video = 'homo_video'

cap = cv.VideoCapture(join(path_to_video_and_timestamp, video))
total_frames = cap.get(7)

cap.set(cv.CAP_PROP_POS_FRAMES, 10998)
ret, img = cap.read()
cv.imshow('frame', img)
cv.waitKey(0)

# for i in range(13):
#     # cap.set(cv.CAP_PROP_POS_FRAMES, ll)
#     ret, img = cap.read()
#     # cv.imshow('frame', img)
#     # cv.waitKey(0)

# c = 0
# while True:
#     ret, img = cap.read()
#     cv.imshow('frame', img)
#     c += 1
#     if c == 79:
#         break
#     if cv.waitKey(100) & 0xFF == ord('q'):
#         break

# cap = cv.VideoCapture(join(path_to_video_and_timestamp, video))
# while True:
#     ret, img = cap.read()
#     cv.imshow('frame', img)
#     print(cap.get(1))
#     if cv.waitKey(10) & 0xFF == ord('q'):
#         break

# rang = range(1,100,3)
# cap = cv.VideoCapture(join(path_to_video_and_timestamp, video))
# for i in rang:
#     cap.set(cv.CAP_PROP_POS_FRAMES, i)
#     ret, img = cap.read()
#     cv.imshow('frame', img)
#     print(cap.get(1))
#     if cv.waitKey(10) & 0xFF == ord('q'):
#         break
