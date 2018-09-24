import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import json
import subprocess
from homography import print_
import cv2 as cv

# with open('exp.json', 'r') as f:
#     experiment = json.load(f)

# x = [v for (k,v) in experiment.items()]

# num_bins = 25
# n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
# plt.show()


path_video = '/home/grigorii/Desktop/momentum_speed/homo_video'

np.fft.fft2