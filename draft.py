from test import *

bc = 10.02
ad = 10
cd = 30
ac = 32.16
bd = 31.02

h = H(bc, ad, cd, ac, bd)
h.find_homography()
h.get_point_transorm()
