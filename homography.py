import cv2
import numpy as np
from math import sqrt
from sympy import solve_poly_system
from sympy.abc import x,y


# solve((y+0.0931)^2+(x-11.8186)^2 = 30.170^2, (y-30.4219)^2+(x-11.8186)^2 = 5.535^2, [x, y])
path_to_test_img = '../test.jpg'
dst_size_height = 2000


def print_(s):
    print('-'*50)
    print(s)
    print('-'*50)

# solve non-linear equations
def find_coords(cd, a_b, b_a):
    x1, y1 = 0, 0
    x2, y2 = 0, cd
    equations = [(y-y1)**2 + (x-x1)**2 - b_a**2, (y-y2)**2 + (x-x2)**2 - a_b**2]
    solutions = solve_poly_system(equations, x, y)
    #TODO точка C может лежать чуть ниже точки B, 
    # тогда solutions везде будут отрицательными - решить момент
    for item in solutions:
        if item[0] >= 0 and item[1] >= 0:
            first = float(item[0])
            second = float(item[1])
            return [first, second]
    raise ValueError('solution to equations is negative')


class Homography:
    def __init__(self, pts_src, pts_real):
        self.pts_src = pts_src
        self.pts_real = pts_real

        # find the border coordinates which constitute a countur
        # these points are corners in the resulting projection
        x_min, x_max = 1000, 0
        y_min, y_max = 1000, 0
        for item in self.pts_real:
            if item[0] < x_min:
                x_min = item[0]
            if item[0] > x_max:
                x_max = item[0]
            if item[1] < y_min:
                y_min = item[1]
            if item[1] > y_max:
                y_max = item[1]

        # get pixel coordinates instead of real coordinates
        resolution_scale = (x_max - x_min) / (y_max - y_min)
        dst_size_width = round(dst_size_height * resolution_scale)
        scale_x = dst_size_width / (x_max - x_min) # coefs for pixels-to-meters transformation
        scale_y = dst_size_height / (y_max - y_min)
        self.scale = (scale_x + scale_y) / 2 #TODO seems a silly step

        # calculate dst_img coordinates automatically
        pts_dst = []
        for item in self.pts_real:
            pts_dst.append([item[0] * scale_x, item[1] * scale_y])
        pts_dst = np.array(pts_dst)

        # print_(type(pts_dst))
        # print_(type(pts_dst[0]))
        # print_(type(pts_dst[0][0]))

        self.h, _ = cv2.findHomography(self.pts_src, pts_dst)

    def get_point_transform(self, src, dst):
        # im_src = cv2.imread(path_to_test_img)
        # im_out = cv2.warpPerspective(im_src, self.h, (int(self.dst_size_width), int(dst_size_height)))
        # cv2.imwrite('test_result_8.jpg', im_out)
        # print()

        # project a point from original image to the projection
        src_proj = np.dot(self.h,np.array([[src[0]],[src[1]],[1]]))
        dst_proj = np.dot(self.h,np.array([[dst[0]],[dst[1]],[1]]))
        #TODO проверка на отрицательные координаты,
        #когда поймали машину вне рамки нашего обзора - продумать момент
        src_proj = src_proj / src_proj[-1]
        dst_proj = dst_proj / dst_proj[-1]
        dist = sqrt((src_proj[0] - dst_proj[0])**2 + (src_proj[1] - dst_proj[1])**2)
        dist_meters = dist / self.scale
        return dist_meters
