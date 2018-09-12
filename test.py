import cv2
import numpy as np
import math


path_to_src = 'photos/1.JPG'
dst_size_height = 2000


def print_(s):
    print('-'*50)
    print(s)
    print('-'*50)


def get_points(file_pts):
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


#TODO find analitycal solution for absolute coordinates

im_src = cv2.imread(path_to_src)

# edge coordinates in meters
a = [9.98, 30.57]
b = [9.99, 0.64]
c = [0, 0]
d = [0, 30]
pts_real = [a, b, c, d]

x_min, x_max = 1000, 0
y_min, y_max = 1000, 0
for item in pts_real:
    if item[0] < x_min:
        x_min = item[0]
    if item[0] > x_max:
        x_max = item[0]
    if item[1] < y_min:
        y_min = item[1]
    if item[1] > y_max:
        y_max = item[1]

resolution_scale = (x_max - x_min) / (y_max - y_min)
dst_size_width = round(dst_size_height * resolution_scale)
scale_x = dst_size_width / (x_max - x_min)
scale_y = dst_size_height / (y_max - y_min)

# calculate dst coordinates automatically
pts_dst = []
for i, item in enumerate(pts_real):
    pts_dst.append([item[0] * scale_x, item[1] * scale_y])
pts_dst = np.array(pts_dst)

# edge points on the photo
A = [2597, 1862]
B = [2327, 1086]
C = [1682, 1061]
D = [1179, 1796]
pts_src =  np.array([A, B, C, D])

h, status = cv2.findHomography(pts_src, pts_dst)

# new_point = np.dot(h,np.array([[2424],[1825],[1]]))
# new_point = new_point/new_point[-1]
# print_("position of the blob on the ground xy plane: {}".format(new_point))

im_out = cv2.warpPerspective(im_src, h, (dst_size_width, dst_size_height))

# x1, y1 = 200, 200
# x2, y2 = 300, 300
# cv2.rectangle(im_src, (x1, y1), (x2, y2), (255,0,0), 10)

# im_out = cv2.resize(im_out, (dst_size_width, dst_size_height))

# corners = np.float32([ [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0] ]).reshape(-1, 1, 2)
# transformedCorners = cv2.perspectiveTransform(corners, M)

cv2.imwrite('res.jpg', im_out)
