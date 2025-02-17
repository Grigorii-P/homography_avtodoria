import numpy as np
import cv2 as cv
import sys
import caffe
from time import time
import json
import os
import pickle
from os.path import join
from homography import Homography, print_
from homography_data import pts_src_, pts_real_
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt


path_to_model = "/home/grigorii/ssd480/talos/python/platedetection/zoo/twins/embeded.prototxt"
path_to_weights = "/home/grigorii/ssd480/talos/python/platedetection/zoo/twins/weights.caffemodel"
path_to_cascade = "/home/grigorii/ssd480/talos/python/platedetection/haar/cascade_inversed_plates.xml"
path_to_video_and_timestamp = '/home/grigorii/Desktop/momentum_speed/video_cruise_control/first_experiment/data'
path_to_targets = '/home/grigorii/Desktop/momentum_speed/experiments/targets'
path_plots = '/home/grigorii/Desktop/momentum_speed/plots'
path_to_homo_img = '../res.jpg'

alphabet = ['1','A','0','3','B','5','C','7','E','9','K','4','X','8','H','2','M','O','P','T','6','Y','@']
minSize_ = (50,10)
maxSize_ = (500,100)
time_ = 0
min_num_appearances = 2 # don't take into account plates that appear < `min_num_appearances` times
frames_threshold = 10 # finalize a plate if doesn't appear this times of frames
total_frames_num = 0
# time_btw_frames_sec = 1/15


def output_plateNumber(img):
    image = cv.resize(img, (160, 32)) / 255.0
    net.blobs['data'].data[...] = image
    net.forward()
    set_index = []

    for i in range(1, 11):
        blob = net.blobs["ip_sm" + str(i)]
        set_index.append(blob.data[0].argmax())

    number = ""
    for index in set_index:
        number += alphabet[index]
    return number


def get_timestamps(path_to_timestamp_file):
    global time_
    with open(path_to_timestamp_file, 'r') as f:
        time_ = {}
        count = 0
        f = open(path_to_timestamp_file, 'r')
        lines = f.read().splitlines()[:-1]
        for line in lines:
            time_[count] = int(line)
            count += 1


def get_time_between_frames(src_frame, dst_frame):
    return abs(time_[dst_frame] - time_[src_frame])


def levenshtein(seq1, seq2):
    '''
    difference distance between two words
    '''
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    # returns the number of changes to be made in order to get two equal words
    return (matrix[size_x - 1, size_y - 1])


def get_momentum_speeds(frames_list, coords, hom):
    global time_
    x, y = [], []
    for i in range(len(frames_list)-1):
        src = coords[i]
        dst = coords[i+1]
        dist_meters = hom.get_point_transform(src, dst)
        t = get_time_between_frames(frames_list[i+1], frames_list[i]) / 1000
        speed = dist_meters / t * 3.6
        x.append(frames_list[i])
        y.append(speed)
        
    median_list = y.copy()
    median_list.sort()
    len_ = len(median_list)
    if len_ % 2 == 0:
        median = (median_list[len_//2] + median_list[len_//2 - 1]) / 2
    else:
        median = median_list[len_//2]
    return [x, y, median]


def get_plots(key, data, speed_overall, speed_av):
    fig, ax = plt.subplots()
    x, y, median = data[0], data[1], data[2]
    axes = plt.gca()
    axes.set_ylim([0,70])
    ax.plot(x, y, label='momentum')
    ax.plot(x, [speed_av] * len(x), label='average')
    ax.plot(x, [median] * len(x), label='median')
    ax.set(xlabel='frames', ylabel='speed')
    ax.legend()
    fig.savefig(join(path_plots, key + '.png'))


def get_track_picture(number, frames_list, plates_coords, hom):
    img = cv.imread(path_to_homo_img)
    for item in frames_list:
        coord = plates_coords[item][number]
        coord_proj = np.dot(hom.h, np.array([[coord[0]], [coord[1]], [1]]))
        coord_proj = coord_proj / coord_proj[-1]
        img = cv.drawMarker(img, tuple(coord_proj[:2]), (0,0,255), markerType=cv.MARKER_TILTED_CROSS, markerSize=20, thickness=3, line_type=cv.LINE_AA)
    cv.imwrite('../track_test/res.jpg', img)
    
    d = 0
    for i in range(len(frames_list) - 1):
        d_inst = hom.get_point_transform(plates_coords[frames_list[i]][number], plates_coords[frames_list[i + 1]][number])
        d += d_inst
        print('      {:.2f}'.format(d_inst))
        print('SUM - {:.2f}'.format(d))


def get_weighted_speed(frames_list, coords, hom):
    global time_
    
    # time duration of the plate lifetime
    t = get_time_between_frames(frames_list[-1], frames_list[0]) / 1000
    
    # overall speed
    src = coords[0]
    dst = coords[-1]
    dist_meters = hom.get_point_transform(src, dst)
    speed_overall = dist_meters / t * 3.6 # km/h

    # average speed
    dist_meters = 0
    for i in range(0, len(coords)-1):
        dist_meters += hom.get_point_transform(coords[i], coords[i+1])
    speed_av = dist_meters / t * 3.6 # km/h

    # return (speed_overall + speed_av) / 2
    return speed_overall, speed_av


def loop_video(video, timestamp, res_file, targets):
    global total_frames_num
    plates_in_frame = {}
    plates_ever_met = {} # {number : [frame_appearances]}
    plates_ever_met_total = {} # additional dict for `visualize_reper_points`
    plates_mean_coords_in_frame = {} # coords of the plate center point
    plates_coords = {} # new dict for calculating speed between any frames of the video
    plate_descriptors = {}
    plates_descriptors_all_frames = {}
    # frames = [] # save frame_counters to visualize_reper_points afterwards
    plots = {}
    
    '''
    # res_file.write('-'*30 + '\n')
    # res_file.write('{} Speed: {}\n'.format(video, targets[video]))
    # res_file.write('-'*30 + '\n')
    '''

    cap = cv.VideoCapture(join(path_to_video_and_timestamp, video))
    get_timestamps(join(path_to_video_and_timestamp, timestamp))
    sift = cv.xfeatures2d.SIFT_create()
    
    # work on each incoming frame
    while True:
        ret, img = cap.read()
        frame_counter = int(cap.get(1)) - 1
        if ret is False:
            break
            raise ValueError('Cannot read a stream')
        
        # get plates via VJ haar
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=minSize_, maxSize=maxSize_)
        
        # second iter for VJ
        if type(plates) is np.ndarray:
            haar_imgs = []
            for (x,y,w,h) in plates:
                img = gray[y:y+h, x:x+w]
                second_iter = plate_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=3, minSize=minSize_, maxSize=maxSize_)
                for item in second_iter:
                    x_new, y_new, w_new, h_new = x + item[0], y + item[1], item[2], item[3]
                    haar_imgs.append([x_new, y_new, w_new, h_new])
            plates = haar_imgs
        for (x,y,w,h) in plates:
            d = 0
            try:
                number = output_plateNumber(gray[y-d:y+h+d, x:x+w])
            except:
                number = output_plateNumber(gray[y:y+h, x:x+w])
            # put each number with its plates into the `plates_in_frame`
            # (there might be more than one plate for each number in one frame)
            if number in plates_in_frame:
                new_rect = plates_in_frame[number]
                new_rect.append((x,y,w,h))
                plates_in_frame[number] = new_rect
            else:
                plates_in_frame[number] = [(x,y,w,h)]
        
        # calculate the mean plate for each number in one frame
        for key in plates_in_frame.keys():
            if key in plates_ever_met:
                frames = plates_ever_met[key]
                frames.append(frame_counter)
                plates_ever_met[key] = frames
            else:
                plates_ever_met[key] = [frame_counter]

            if key in plates_ever_met_total:
                frames = plates_ever_met_total[key]
                frames.append(frame_counter)
                plates_ever_met_total[key] = frames
            else:
                plates_ever_met_total[key] = [frame_counter]

            coords = plates_in_frame[key]
            x_sum, y_sum, w_sum, h_sum, count = 0, 0, 0, 0, 0
            for item in coords:
                x_sum += item[0]
                y_sum += item[1]
                w_sum += item[2]
                h_sum += item[3]
                count += 1
            x_av, y_av, w_av, h_av = x_sum // count, y_sum // count, w_sum // count, h_sum // count #av - average
            plates_mean_coords_in_frame[key] = [x_av + w_av // 2, y_av + h_av // 2]
            kp, des = sift.detectAndCompute(gray[y_av:y_av + h_av, x_av:x_av + w_av], None)
            plate_descriptors[key] = [kp, des, (x_av, y_av, w_av, h_av)]
           
            '''
            # save frames with a crossing on a plate
            crossing()
            '''
            
        plates_coords[frame_counter] = plates_mean_coords_in_frame.copy()
        plates_descriptors_all_frames[frame_counter] = plate_descriptors.copy()

        # calculate the average speed between first and the last frames of the number 
        # that didn't appear after `frames_threshold` times (finalize it)
        plates_to_del = []
        for key in plates_ever_met:
            last = plates_ever_met[key][-1]
            if frame_counter - last >= frames_threshold:
                # if the plate appeared less than `min_num_appearances` times, 
                # before the last 25 frames, don't take it into account
                if len(plates_ever_met[key]) < min_num_appearances:
                    plates_to_del.append(key)
                    continue
                '''
                # _, speed_av = get_weighted_speed(key, plates_ever_met[key], plates_coords, hom)
                # output = get_momentum_speeds(key, plates_ever_met[key], plates_coords, hom)
                # speed_median = output[-1]
                # res_file.write('{} {:.1f} {:.1f}\n'.format(key, speed_av, speed_median))
                # plots[key] = get_momentum_speeds(key, plates_ever_met[key], plates_coords, hom)
                # if len(plots[key][0]) > 1: # we can't plot only one point, we need more than one
                #     speed_overall, speed_av = get_weighted_speed(key, plates_ever_met[key], plates_coords, hom)
                #     get_plots(key, plots[key], speed_overall, speed_av)
                '''
                plates_to_del.append(key)
        for item in plates_to_del:
            del plates_ever_met[item]
        
        total_frames_num += 1
        plates_in_frame.clear()
        plates_mean_coords_in_frame.clear()
        plate_descriptors.clear()
        
        print('frames - {}, total - {}'.format(frame_counter, total_frames_num))
        # if total_frames_num == 150:
        #     break
            # return False, None, None
    
    cap.release()
    return True, plates_ever_met_total, plates_descriptors_all_frames


def visualize_reper_points(path_to_imgs, plates_ever_met_total, plates_descriptors_all_frames):
    # numbers that we want to trace
    os.system('mkdir ' + path_to_imgs)
    target = 'A283CO716@'
    targets = ['A283CO716@']
    points = []
    # get frames sequences for each of the above mentioned numbers
    # for target in targets:
    #     frames[target] = plates_ever_met_total[target]
    if target in plates_ever_met_total:
        frames = plates_ever_met_total[target]
    else:
        return 0, 0

    # im_size = (1920, 1406)
    # params for kd trees
    FLANN_INDEX_KDTREE = 1 # might be 0 as well
    trees = 5
    checks = 50
    # params for finding the closest neighbours
    k = 2
    ratio = 0.75
    min_match_count = 4
    # homography param
    # E.g. if dst_points coordinates are measured in pixels with pixel-accurate precision, 
    # it makes sense to set this parameter somewhere in the range ~1..3
    ransacReprojThreshold = 3
    
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
    search_params = dict(checks = checks)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    cap = cv.VideoCapture(join(path_to_video_and_timestamp, video))
    frame_count_all = 0

    # for each number run a video again and again
    for target in targets:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        
        k_prev, d_prev, (x_av, y_av, w_av, h_av) = plates_descriptors_all_frames[frames[0]][target]
        point_to_be_transformed = np.float32([w_av // 2, h_av // 2]).reshape(-1,1,2)
        
        while cap.get(1) < frames[0]:
            _, img = cap.read()

        _, first_img = cap.read()
        first_img = cv.drawMarker(first_img, (x_av + w_av // 2, y_av + h_av // 2), (34,139,34), markerType=cv.MARKER_TILTED_CROSS, markerSize=10, thickness=1, line_type=cv.LINE_AA)
        cv.imwrite(join(path_to_imgs, str(frame_count_all) + '.jpg'), first_img)
        frame_count_all += 1
        points.append((x_av + w_av // 2, y_av + h_av // 2))

        for frame in frames[1:]:
            while True:
                ret, img = cap.read()
                if cap.get(1)-1 == frame or ret is False:
                    break

            k_next, d_next, (x_av, y_av, w_av, h_av) = plates_descriptors_all_frames[frame][target]
            matches = flann.knnMatch(d_prev, d_next, k=k)
            good = []
            for m, n in matches:
                if m.distance < ratio * n.distance:
                    good.append([m])

            if len(good) > min_match_count:
                src_pts = np.float32([ k_prev[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ k_next[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
                local_hom, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransacReprojThreshold)
                dst = cv.perspectiveTransform(point_to_be_transformed, local_hom)
                if len(good) > 7:
                    img = cv.drawMarker(img, (int(x_av + dst[0][0][0]), int(y_av + dst[0][0][1])), (34,139,34), markerType=cv.MARKER_TILTED_CROSS, markerSize=10, thickness=1, line_type=cv.LINE_AA)
                else:
                    img = cv.drawMarker(img, (int(x_av + dst[0][0][0]), int(y_av + dst[0][0][1])), (0,0, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=10, thickness=1, line_type=cv.LINE_AA)
                cv.imwrite(join(path_to_imgs, str(frame_count_all) + '.jpg'), img)
                k_prev, d_prev = k_next, d_next
                point_to_be_transformed = dst
                frame_count_all += 1
                points.append((int(x_av + dst[0][0][0]), int(y_av + dst[0][0][1])))
                print("Num matches - %d/%d" % (len(good), min_match_count))
            else:
                print("NOT ENOUGH MATCHES - %d/%d" % (len(good), min_match_count))
                cv.imwrite(join(path_to_imgs, 'NOT ENOUGH MATCHES' + str(frame_count_all) + '.jpg'), img)
    cap.release()

    if len(points) > 1:
        _, speed_av = get_weighted_speed(frames, points, hom)
        output = get_momentum_speeds(frames, points, hom)
        speed_median = output[-1]
        return speed_av, speed_median
    else:
        return 0, 0


if __name__ == "__main__":
    # res_file = open('../video_cruise_control/second_experiment/temp/speed_file', 'w')
    # with open(path_to_targets, 'r') as targets_file:
    #     lines = targets_file.read().splitlines()
    # targets = dict((x.split(' ')[0], float(x.split(' ')[1])) for x in lines)
    
    hom = Homography(np.array(pts_src_), np.array(pts_real_))
    plate_cascade = cv.CascadeClassifier(path_to_cascade)
    caffe.set_mode_cpu()
    net = caffe.Net(path_to_model, path_to_weights, caffe.TEST)

    # videos = []
    # timestamps = []
    # files = os.listdir(path_to_video_and_timestamp)
    # for f in files:
    #     if not f.endswith('_timestamps') and f.endswith('mp4'):
    #         videos.append(f)
    #         timestamps.append(f + '_timestamps')
    # v_t = dict(zip(videos, timestamps))
    
    # v_t = {'1.mp4':'', '3.mp4':'', '5.mp4':'', '7.mp4':'', '9.mp4':'', '11.mp4':''}
    # v_t = {'2.mp4':'', '4.mp4':'', '6.mp4':'', '8.mp4':'', '10.mp4':'', '12.mp4':''}
    v_t = {'regid_1538566412031_ffv1_59':'regid_1538566412031_ffv1_59_timestamps'}

    for video, timestamp in v_t.items():
        flag, plates_ever_met, plates_descriptors_all_frames = loop_video(video, timestamp, None, None) # (video, timestamp, res_file, targets)
        path_to_imgs = join('../video_cruise_control/first_experiment/temp', video)
        speed_av, speed_median = visualize_reper_points(path_to_imgs, plates_ever_met, plates_descriptors_all_frames)
        # res_file.write('{} {:.1f} {:.1f}\n'.format(video, speed_av, speed_median))

        if not flag:
            break

    # res_file.close()
