import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import os
from time import time
from os.path import join
from annoy import AnnoyIndex
from itertools import combinations_with_replacement as cwr
import itertools

path_to_imgs = '../repers/plates1000'
path_to_save = '../repers/binary'

#TODO нормализация, выравнивание изображения

# _, cnts, _ = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# for c in cnts:
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#     x, y, w, h = cv2.boundingRect(approx)
#     cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 1)
#     # if h >= 15:
#     #     # if height is enough
#     #     # create rectangle for bounding
#     #     rect = (x, y, w, h)
#     #     rects.append(rect)
#     #     cv2.rectangle(roi_copy, (x, y), (x+w, y+h), (0, 255, 0), 1)

# # cv2.rectangle(img, (5, 5), (25, 25), (100, 100, 100), 1)
# cv2.imwrite(path_to_save, img)

path_to_imgs = '/home/grigorii/Desktop/momentum_speed/repers/annoy_test'


imgs_size = (150, 30)
search_w, search_h = 10, 10
num_trees = 10

im_src = cv2.imread(join(path_to_imgs, 'B418EY716@_13.jpg'), 0)
im_dst = cv2.imread(join(path_to_imgs, 'B418EY716@_14.jpg'), 0)
im_src = cv2.resize(im_src, imgs_size)
im_dst = cv2.resize(im_dst, imgs_size)
w = im_src.shape[1]
h = im_src.shape[0]

f = search_w * search_h
t = AnnoyIndex(f)
offset = w * h // f
scales = [] #TODO нужнор подобрать так, чтобы нашим окном по-прежнем можно было пройтись по ресайзнутой картинке

# t0 = time()
# num = 1000
# for i in range(num):
ind_1, ind_2 = 0, offset
ind_1_list = []
ind_2_list = []
distances = {}
for scale in scales:
    new_w = int(w * scale)
    new_h = int(h * scale)
    im_dst = cv2.resize(im_dst, (new_w, new_h))
    for i in range(0, w, search_w):
        for j in range(0, h, search_h):
            t.add_item(ind_1, im_src[j:j+search_h, i:i+search_w].flatten())
            ind_1_list.append(ind_1)
            ind_1 += 1
            # temp = im_dst[j:j+search_h, i:i+search_w]
            # temp = cv2.resize(temp, (100, 100))
            # cv2.imshow('temp', temp)
            # cv2.waitKey(0)
    for i in range(0, new_w, search_w):
        for j in range(0, new_h, search_h):
            t.add_item(ind_2, im_dst[j:j+search_h, i:i+search_w].flatten())
            ind_2_list.append(ind_2)
            ind_2 += 1

t.build(num_trees)
# t.save('../repers/annoy_test/test.ann')

combinations = list(itertools.product(ind_1_list, ind_2_list))
for combo in combinations:
    dist = t.get_distance(combo[0], combo[1])
    distances[combo] = dist
combs_sorted_by_dist = sorted(distances.items(), key=lambda kv: kv[1])
# best = combs_sorted_by_dist[-1]
# print('average time - {} sec'.format((time()-t0)/num))

best = combs_sorted_by_dist[-3:]
for i in range(len(best)):
    im_1 = t.get_item_vector(best[i][0][0])
    vec = np.reshape(im_1, (search_h, search_w))
    img = cv2.resize(vec, (100, 100))
    img = img.astype(dtype=np.uint8)
    cv2.imwrite(join(path_to_imgs + '/test', 'first_' + str(i) + '.jpg'), img)

    im_2 = t.get_item_vector(best[i][0][1])
    vec = np.reshape(im_2, (search_h, search_w))
    img = cv2.resize(vec, (100, 100))
    img = img.astype(dtype=np.uint8)
    cv2.imwrite(join(path_to_imgs + '/test', 'second_' + str(i) + '.jpg'), img)
