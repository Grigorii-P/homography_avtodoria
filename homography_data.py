import numpy as np


# RELATIVE COORDINATES: A(1,1), B(1.0), C(0,0), D(0,1)

#
## First experiment
#
# coordinates of the real points
a = [11.8186, 30.4219]
b = [11.8186, -0.0931]
c = [0, 0]
d = [0.3212, 16.2902]
e = [6.3485, 29.5769]
f = [6.32664, 22.3198]
g = [6.31042, 14.2978]
h = [6.33798, 2.88398]
# coordinates of the corresponding points on the image
A = [1728, 786]
B = [1865, 49]
C = [615, 65]
D = [140, 325]
E = [560, 731]
F = [844, 461]
G = [1048, 263]
H = [1242, 83]

pts_src_ =  np.array([A, B, C, D, E, F, G, H])
pts_real_ =  np.array([a, b, c, d, e, f, g, h])

#
## Cruise control experiment
#

