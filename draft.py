import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import json

# with open('exp.json', 'r') as f:
#     experiment = json.load(f)

# x = [v for (k,v) in experiment.items()]

# num_bins = 25
# n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
# plt.show()

dist_meters = 5
t = 1
speed_overall = dist_meters / t * 3.6 # km/h

# average speed
dist_meters = 0
dist_meters = 6
speed_av = dist_meters / t * 3.6 # km/h

print(speed_overall, speed_av)