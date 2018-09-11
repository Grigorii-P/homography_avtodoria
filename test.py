import cv2
import numpy as np
import math
from sympy import Symbol, nsolve
import sympy
import mpmath


path_to_src = 'photos/1.JPG'
path_to_new_src = 'photos/1.JPG'

dst_size = (round(4000 / 3.057), 4000)


def get_coords(x1, y1, x2, y2, a, c):
    x = Symbol('x')
    y = Symbol('y')

    f1 = (y-y1)**2+(x-x1)**2 - c**2 
    f2 = (y-y2)**2+(x-x2)**2 - a**2

    return list(nsolve((f1, f2), (x, y), (10, 20)))


def get_points():
    with open(file_pts) as f:
        content = f.readlines()
    content = [x.replace('\n', '') for x in content]

    pts_src = np.zeros(shape=(8))
    for i, item in enumerate(content[0].split(' ')):
        pts_src[i] = int(item)
    pts_src = np.reshape(pts_src, newshape=(4, 2))

    pts_dst = np.zeros(shape=(8))
    for i, item in enumerate(content[1].split(' ')):
        pts_dst[i] = int(item)
    pts_dst = np.reshape(pts_dst, newshape=(4, 2))

    return pts_src, pts_dst


im_src = cv2.imread(path_to_src)
im_new_src = cv2.imread(path_to_new_src)

# order: A-B-C-D
pts_src =  np.array([[2597, 1862], [2327, 1086], [1682, 1061], [1179, 1796]])
# pts_dst =  np.array([[1497, 0], [1500, 1959], [0, 2000], [0, 37]])
pts_dst =  np.array([[4024, 0], [4032, 2962], [0, 3024], [0, 56]])

h, status = cv2.findHomography(pts_src, pts_dst)
im_out = cv2.warpPerspective(im_new_src, h, (im_src.shape[1], im_src.shape[0]))
# im_out = cv2.warpPerspective(im_new_src, h, (im_src.shape[0], im_src.shape[1]))
im_out = cv2.resize(im_out, (dst_size[0], dst_size[1]))
# im_out = cv2.resize(im_out, (dst_size[1], dst_size[0]))

# corners = np.float32([ [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0] ]).reshape(-1, 1, 2)
# transformedCorners = cv2.perspectiveTransform(corners, M)

# cv2.imshow("Warped Source Image", im_out)
cv2.imwrite('res.jpg', im_out)
# cv2.waitKey(0)
