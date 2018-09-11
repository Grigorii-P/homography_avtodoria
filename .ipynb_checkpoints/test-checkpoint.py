import cv2
import numpy as np

path_to_src = '/home/grigorii/Downloads/src2.jpg'
path_to_dst = '/home/grigorii/Downloads/dst.jpg'
file_pts = 'src_dst_points'


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


pts_src, pts_dst = get_points()

im_src = cv2.imread(path_to_src)
# pts_src = np.array([[2206, 1329], [2705, 1515], [1646, 1493], [2106, 1842]])

im_dst = cv2.imread(path_to_dst)
# pts_dst = np.array([[1420, 1209], [2641, 1381], [1081, 2505], [2477, 3013]])

h, status = cv2.findHomography(pts_src, pts_dst)
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
im_out = cv2.resize(im_out, (750, 1000))

# corners = np.float32([ [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0] ]).reshape(-1, 1, 2)
# transformedCorners = cv2.perspectiveTransform(corners, M)

# cv2.imshow("Warped Source Image", im_out)
cv2.imwrite('res.jpg', im_out)
cv2.waitKey(0)
