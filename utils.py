import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import cv2 as cv
import os
from os.path import join
from homography import Homography
from homography_data import *


def errors_histogram():
    path_targets = 'TARGETS'
    path_result = 'RESULT_NEW'

    # res = open(path_result, 'r')

    targets = {}
    results = {}

    with open(path_targets) as f:
        content = f.readlines()
        content = [x[:-1] for x in content]
        for item in content:
            targets[item.split(' ')[0]] = int(item.split(' ')[1])

    with open(path_result) as f:
        content = f.readlines()
        content = [x[:-1] for x in content]
        for item in content:
            results[item.split(' ')[0].translate({ord('@'):None})] = float(item.split(' ')[2])


    errors = []
    for k in targets:
        if k in results:
            errors.append(results[k] - targets[k])


    num_bins = 70
    n, bins, patches = plt.hist(errors, num_bins, facecolor='blue', alpha=0.5)
    plt.show()


def get_point_transform():
    hom = Homography(np.array(pts_src_), np.array(pts_real_))
    # dist = hom.get_point_transform([1136, 184], [942, 363])
    A = [1701, 746]
    B = [1851, 43]
    dist = hom.get_point_transform(A, B)
    print()


def binary_threshold():
    path_to_imgs = '../repers/plates1000'
    path_to_save = '../repers/binary'

    plates = os.listdir(path_to_imgs)
    c = 0
    for plate in plates:
        img = cv2.imread(join(path_to_imgs, plate), 0)
        th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
        cv2.imwrite(join(path_to_save, str(c) + '.jpg'), th3)
        c += 1
        # (thresh, im_bw) = cv.threshold(img, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        # th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
        # th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, r)


# https://github.com/spotify/annoy
def annoy_spotify():
    path_to_imgs = '/home/grigorii/Desktop/momentum_speed/repers/annoy_test'

    imgs_size = (150, 30)
    search_w, search_h = 15, 15
    num_trees = 20

    im_src = cv2.imread(join(path_to_imgs, 'A402AC116@_106.jpg'), 0)
    im_dst = cv2.imread(join(path_to_imgs, 'A402AC116@_109.jpg'), 0)
    im_src = cv2.resize(im_src, imgs_size)
    im_dst = cv2.resize(im_dst, imgs_size)
    w = im_src.shape[1]
    h = im_src.shape[0]

    f = search_w * search_h
    t = AnnoyIndex(f)
    offset = 1000
    # scales = [0.8, 0.9, 1, 1.1, 1.2]
    scales = [1]

    # t0 = time()
    # num = 1000
    # for i in range(num):
    ind_1, ind_2 = 0, offset
    ind_1_list = []
    ind_2_list = []
    distances = {}
    for i in range(0, w, search_w):
            for j in range(0, h, search_h):
                t.add_item(ind_1, im_src[j:j+search_h, i:i+search_w].flatten())
                ind_1_list.append(ind_1)
                ind_1 += 1
                # temp = im_dst[j:j+search_h, i:i+search_w]
                # temp = cv2.resize(temp, (100, 100))
                # cv2.imshow('temp', temp)
                # cv2.waitKey(0)
    for scale in scales:
        new_w = int(w * scale)
        new_h = int(h * scale)
        im_dst_res = cv2.resize(im_dst, (new_w, new_h))
        for i in range(0, new_w, search_w):
            for j in range(0, new_h, search_h):
                item = im_dst_res[j:j+search_h, i:i+search_w].flatten()
                if item.size < f:
                    item = np.append(item, [255] * (f - item.size))
                t.add_item(ind_2, item)
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


def crossing():
    if key == 'A283CO716@':
        coord = plates_mean_coords_in_frame[key]
        gray = cv.drawMarker(gray, (coord[0], coord[1]), (0,0,255), markerType=cv.MARKER_TILTED_CROSS, markerSize=15, thickness=2, line_type=cv.LINE_AA)
        cv.imwrite('../track_test/' + str(key) + '_' + str(c) + '.jpg', gray)
        
        font                   = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2
        
        cv.putText(img,'Hello World!', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        c += 1


def calc_descriptors_and_similarity():
    path_to_imgs = '../repers/annoy_test'
    path_to_save = '../repers/binary'

    img_size = (450, 90)
    k = 2
    ratio = 0.5
    MIN_MATCH_COUNT = 5
    templates = os.listdir(path_to_imgs)
    templates = [x for x in templates if x.endswith('.jpg') and '@' in x]

    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher()

    im_src = cv2.imread(join(path_to_imgs, templates[0]), 0)
    h, w = im_src.shape
    k_prev, d_prev = sift.detectAndCompute(im_src, None)
    src_point = np.float32([[w // 2, h // 2]]).reshape(-1,1,2)

    im_src = cv2.drawMarker(im_src, (w // 2, h // 2), (0,0,255), markerType=cv.MARKER_TILTED_CROSS, markerSize=10, thickness=1, line_type=cv.LINE_AA)
    im_src = cv2.resize(im_src, img_size)
    cv2.imwrite(join(path_to_imgs, 'first.jpg'), im_src)

    c = 1
    for i in range(1, len(templates), 1):
        im_dst = cv2.imread(join(path_to_imgs, templates[i]), 0)
        # im_dst = cv2.resize(im_dst, img_size)
        k_next, d_next = sift.detectAndCompute(im_dst, None)

        matches = bf.knnMatch(d_prev, d_next, k=k)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < ratio * n.distance:
                good.append([m])
        
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([ k_prev[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ k_next[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)

            local_hom, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            h, w = im_src.shape
            dst = cv.perspectiveTransform(src_point, local_hom)

            im_dst = cv2.drawMarker(im_dst, (int(dst[0][0][0]), int(dst[0][0][1])), (0,0,255), markerType=cv.MARKER_TILTED_CROSS, markerSize=10, thickness=1, line_type=cv.LINE_AA)
            im_dst = cv2.resize(im_dst, img_size)
            cv2.imwrite(join(path_to_imgs, str(c) + '.jpg'), im_dst)

            k_prev, d_prev = k_next, d_next
            src_point = dst
            c += 1
        else:
            print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))

        # # cv2.drawMatchesKnn expects list of lists as matches.
        # img3 = cv2.drawMatchesKnn(im_src, kp1, im_dst, kp2, good, None, flags=2)
        # # cv2.drawMatchesKnn

        # plt.imshow(img3)
        # plt.title(templates[i] + '_' + templates[i+1])
        # plt.show()

calc_descriptors_and_similarity()
