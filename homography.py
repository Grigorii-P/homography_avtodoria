import cv2
import numpy as np
import math
from sympy import solve_poly_system
from sympy.abc import x,y


# COORDINATES: A(1,1), B(1.0), C(0,0), D(0,1)
path_to_src = 'photos/1.JPG'
dst_size_height = 2000

# edge points on the photo
A = [2597, 1862]
B = [2327, 1086]
C = [1682, 1061]
D = [1179, 1796]
pts_src =  np.array([A, B, C, D])


def print_(s):
    print('-'*50)
    print(s)
    print('-'*50)


def find_coords(cd, a_b, b_a):
    x1, y1 = 0, 0
    x2, y2 = 0, cd
    equations = [(y-y1)**2 + (x-x1)**2 - b_a**2, (y-y2)**2 + (x-x2)**2 - a_b**2]
    solutions = solve_poly_system(equations, x, y)
    for item in solutions:
        if item[0] >= 0 and item[1] >= 0:
            first = float(item[0])
            second = float(item[1])
            return [first, second]
    raise ValueError('solution to equations is negative')


class H:
    def __init__(self, bc_, ad_, cd_, ac_, bd_):
        # lenghts between points in meters
        self.bc = bc_
        self.ad = ad_
        self.cd = cd_
        self.ac = ac_
        self.bd = bd_

    def find_homography(self):
        # find a and b coordinates
        a = find_coords(self.cd, self.ad, self.ac)
        b = find_coords(self.cd, self.bd, self.bc)
        c = [0, 0]
        d = [0, self.cd]
        pts_real = [a, b, c, d]

        # find the border coordinates which constitute a countur
        # these points are corners in the resulting projection
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
        self.dst_size_width = round(dst_size_height * resolution_scale)
        scale_x = self.dst_size_width / (x_max - x_min) # coefs for pixels-to-meters transformation
        scale_y = dst_size_height / (y_max - y_min)

        # calculate dst_img coordinates automatically
        pts_dst = []
        for item in pts_real:
            pts_dst.append([item[0] * scale_x, item[1] * scale_y])
        pts_dst = np.array(pts_dst)

        # print_(type(pts_dst))
        # print_(type(pts_dst[0]))
        # print_(type(pts_dst[0][0]))

        self.h, status = cv2.findHomography(pts_src, pts_dst)

    def get_point_transorm(self):
        im_src = cv2.imread(path_to_src)

        # project a point from original image to the projection
        new_point = np.dot(self.h,np.array([[1680],[1561],[1]]))
        new_point = new_point/new_point[-1]
        print_('{}'.format(new_point[:2]))

        im_out = cv2.warpPerspective(im_src, self.h, (self.dst_size_width, dst_size_height))
        cv2.imwrite('res.jpg', im_out)
