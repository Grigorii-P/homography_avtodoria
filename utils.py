import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from homography import *
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