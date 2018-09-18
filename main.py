#Первичное обнаружение ГРЗ в видеопотоке и распознавание её содержимого
import numpy as np
import cv2 as cv
import sys
import caffe
import time
from os.path import join
from homography import *


path_to_model = "/home/grigorii/ssd480/talos/python/platedetection/zoo/twins/embeded.prototxt"
path_to_weights = "/home/grigorii/ssd480/talos/python/platedetection/zoo/twins/weights.caffemodel"
path_to_cascade = "/home/grigorii/ssd480/talos/python/platedetection/haar/cascade_inversed_plates.xml"
path_to_video = '/home/grigorii/Desktop/primary_search/video-1.mp4'
path_to_save_frames = '/home/grigorii/Desktop/homography/test_frames'

alphabet = ['1','A','0','3','B','5','C','7','E','9','K','4','X','8','H','2','M','O','P','T','6','Y','@']
minSize_ = (50,10)
maxSize_ = (200,40)
time_per_frame = 0
#TODO добавить шаг пересчета скоростей (напр, брать каждый второй фрейм или десятый и тд)


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


def target_list(path_to_file):
    with open(path_to_file) as f:
        content = f.readlines()
        content = [x[:-1] for x in content]
        content = [x+'@'*(10-len(x)) for x in content]
        return content


def get_curr_speed(plates_coords, frame, h):
    global time_per_frame
    curr_cars = plates_coords[frame].keys()
    prev_cars = plates_coords[frame - 1].keys()
    plates_in_both_frames = list(set(curr_cars).intersection(prev_cars))
    if plates_in_both_frames:
        for number in plates_in_both_frames:
            src = plates_coords[frame - 1][number]
            dst = plates_coords[frame][number]
            dist_meters = h.get_point_transform(src, dst)
            speed = dist_meters / time_per_frame_sec * 3.6 # km/h
            print('{} - {} km/h'.format(number, speed))


if __name__ == "__main__":
    frame_counter = 0
    plates_mean_coords_in_frame = {} # coords of the plate center point
    plates_coords = {}
    current_speed = {}

    # distances between points in meters
    bc = 5
    ad = 6
    cd = 5.5
    ac = 7.8
    bd = 7.1
    # coordinates of the corresponding points on the image
    A = [1659, 680]
    B = [1837, 249]
    C = [652, 239]
    D = [15, 605]
    pts_src =  np.array([A, B, C, D])
    hom = H(bc, ad, cd, ac, bd, pts_src)
    hom.find_homography()

    cap = cv.VideoCapture(path_to_video)
    cap.set(cv.CAP_PROP_POS_AVI_RATIO, 1)
    video_length_msec = cap.get(cv.CAP_PROP_POS_MSEC)
    frames_num = cap.get(cv.CAP_PROP_POS_FRAMES)
    time_per_frame_sec = video_length_msec / frames_num / 1000 # to get sec instead of msec
    
    plate_cascade = cv.CascadeClassifier(path_to_cascade)
    caffe.set_mode_cpu()
    net = caffe.Net(path_to_model, path_to_weights, caffe.TEST)

    # work on each incoming frame
    #TODO костыль, без этой повторной строчки видео не откроется
    cap = cv.VideoCapture(path_to_video)
    plates_in_frame = {}
    while True:
        ret, img = cap.read()
        if ret is False:
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
        
        if frame_counter == 139:
            print()
        for i, (x,y,w,h) in enumerate(plates):
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
            num = plates_in_frame[key]
            x_av, y_av, w_av, h_av, count = 0, 0, 0, 0, 0 # av - average
            for item in num:
                x_av += item[0]
                y_av += item[1]
                w_av += item[2]
                h_av += item[3]
                count += 1           
            plates_mean_coords_in_frame[key] = [round(x_av / count + (w_av / count) / 2), 
                                                round(y_av / count + (h_av / count) / 2)]
        
        if frame_counter == 57:
            ass=5
        # new dict for calculating speed between any frames of the video
        plates_coords[frame_counter] = plates_mean_coords_in_frame.copy()

        if frame_counter > 0:
            if bool(plates_coords):
                print('-'*50)
                print('frame {}\n'.format(frame_counter))
                get_curr_speed(plates_coords, frame_counter, hom)
                print('-'*50)

        frame_counter += 1
        plates_in_frame.clear()
        plates_mean_coords_in_frame.clear()
        
    # cv.destroyAllWindows()
