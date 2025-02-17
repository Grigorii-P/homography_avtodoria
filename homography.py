import cv2
import numpy as np
from math import sqrt
from sympy import solve_poly_system
from sympy.abc import x,y


# solve((y+0.0931)^2+(x-11.8186)^2 = 30.170^2, (y-30.4219)^2+(x-11.8186)^2 = 5.535^2, [x, y])
path_to_test_img = '../test.jpg'
dst_size_height = 2000

# The maximum allowed reprojection error to treat a point pair as an inlier. 
# The parameter is only used in RANSAC-RHO-based homography estimation. 
ransacReprojThreshold = 0.4 # 40cm


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
    # тогда solutions везде будут отрицательными
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

        # find the border coordinates which constitute a contour;
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

        self.homs = []
        params = [cv2.RANSAC, cv2.RHO, cv2.LMEDS]
        nums = [0.1,0.2,0.3,0.4,0.5]
        nums = [i * 1000 for i in nums]
        for par in params:
            for n in nums:
                self.h, _ = cv2.findHomography(self.pts_src, pts_dst, par, n)
                self.homs.append([self.h, par, n])
        # self.h, _ = cv2.findHomography(self.pts_src, pts_dst, cv2.RANSAC, 5.0)
        # cv2.findHomogra

    def get_point_transform(self, src, dst):
        #TODO use perspectiveTransform() instead
        # project a point from original image to the projection
        src_proj = np.dot(self.h,np.array([[src[0]],[src[1]],[1]]))
        dst_proj = np.dot(self.h,np.array([[dst[0]],[dst[1]],[1]]))
        src_proj = src_proj / src_proj[-1]
        dst_proj = dst_proj / dst_proj[-1]
        dist = sqrt((src_proj[0] - dst_proj[0])**2 + (src_proj[1] - dst_proj[1])**2)
        dist_meters = dist / self.scale
        return dist_meters
    
    # for experiments in draft.py only
    def get_point_transform_2(self, h, src, dst):
        src_proj = np.dot(h,np.array([[src[0]],[src[1]],[1]]))
        dst_proj = np.dot(h,np.array([[dst[0]],[dst[1]],[1]]))
        src_proj = src_proj / src_proj[-1]
        dst_proj = dst_proj / dst_proj[-1]
        dist = sqrt((src_proj[0] - dst_proj[0])**2 + (src_proj[1] - dst_proj[1])**2)
        dist_meters = dist / self.scale
        return dist_meters


