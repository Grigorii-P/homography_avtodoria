#Первичное обнаружение ГРЗ в видеопотоке и распознавание её содержимого
import numpy as np
import cv2 as cv
import sys
import caffe
import time
from os.path import join
from homography import Homography, print_
import json
from moviepy.editor import VideoFileClip


path_to_model = "/home/grigorii/ssd480/talos/python/platedetection/zoo/twins/embeded.prototxt"
path_to_weights = "/home/grigorii/ssd480/talos/python/platedetection/zoo/twins/weights.caffemodel"
path_to_cascade = "/home/grigorii/ssd480/talos/python/platedetection/haar/cascade_inversed_plates.xml"
path_to_video = '/home/grigorii/Desktop/momentum_speed/homo_video'

alphabet = ['1','A','0','3','B','5','C','7','E','9','K','4','X','8','H','2','M','O','P','T','6','Y','@']
minSize_ = (50,10)
maxSize_ = (200,40)
time_per_frame_sec = 0
num_frames_in_video = 6540
min_num_appearances = 2 # don't take into account plates that appear < `min_num_appearances` times
frames_threshold = 10 # finalize a plate if doesn't appear this times of frames


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


def get_curr_speed(plates_coords, frame, hom):
    global time_per_frame_sec
    curr_cars = plates_coords[frame].keys()
    prev_cars = plates_coords[frame - 1].keys()
    plates_in_both_frames = list(set(curr_cars).intersection(prev_cars))
    if plates_in_both_frames:
        for number in plates_in_both_frames:
            src = plates_coords[frame - 1][number]
            dst = plates_coords[frame][number]
            dist_meters = hom.get_point_transform(src, dst)
            speed = dist_meters / time_per_frame_sec * 3.6 # km/h
            print('{} - {} km/h'.format(number, speed))


def get_weighted_speed(number, frames_list, plates_coords, hom):
    global time_per_frame_sec
    
    # time duration of the plate lifetime
    t = (frames_list[-1] - frames_list[0]) * time_per_frame_sec
    
    # overall speed
    src = plates_coords[frames_list[0]][number]
    dst = plates_coords[frames_list[-1]][number]
    dist_meters = hom.get_point_transform(src, dst)
    speed_overall = dist_meters / t * 3.6 # km/h

    # average speed
    dist_meters = 0
    coords = []
    for frame in frames_list:
        coords.append(plates_coords[frame][number])
    for i in range(0, len(coords)-1):
        dist_meters += hom.get_point_transform(coords[i], coords[i+1])
    speed_av = dist_meters / t * 3.6 # km/h

    return (speed_overall + speed_av) / 2
        

if __name__ == "__main__":
    frame_counter = 0
    plates_in_frame = {}
    plates_ever_met = {} # {number : [frame_appearances]}
    plates_mean_coords_in_frame = {} # coords of the plate center point
    plates_coords = {} # new dict for calculating speed between any frames of the video

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
    pts_src =  np.array([A, B, C, D, E, F, G, H])
    pts_real =  np.array([a, b, c, d, e, f, g, h])
    hom = Homography(pts_src, pts_real)

    clip = VideoFileClip(path_to_video)
    time_per_frame_sec = clip.duration / num_frames_in_video
    
    plate_cascade = cv.CascadeClassifier(path_to_cascade)
    caffe.set_mode_cpu()
    net = caffe.Net(path_to_model, path_to_weights, caffe.TEST)

    # work on each incoming frame
    cap = cv.VideoCapture(path_to_video)
    res_file = open('RESULT', 'w')
    while True:
        ret, img = cap.read()
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
            if key in plates_ever_met:
                frames = plates_ever_met[key]
                frames.append(frame_counter)
                plates_ever_met[key] = frames
            else:
                plates_ever_met[key] = [frame_counter]

            coords = plates_in_frame[key]
            x_av, y_av, w_av, h_av, count = 0, 0, 0, 0, 0 # av - average
            for item in coords:
                x_av += item[0]
                y_av += item[1]
                w_av += item[2]
                h_av += item[3]
                count += 1

            plates_mean_coords_in_frame[key] = [round(x_av / count + (w_av / count) / 2), 
                                                round(y_av / count + (h_av / count) / 2)]
        
        plates_coords[frame_counter] = plates_mean_coords_in_frame.copy()

        # print the speed at each frame
        # if frame_counter > 0:
        #     if bool(plates_coords):
        #         print('-'*50)
        #         print('frame {}\n'.format(frame_counter))
        #         get_curr_speed(plates_coords, frame_counter, hom)
        #         print('-'*50)

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
                speed = get_weighted_speed(key, plates_ever_met[key], plates_coords, hom)
                # print_('frame - {}, {} - {}'.format(frame_counter, key, speed))
                res_file.write('{} {}\n'.format(key, speed))
                plates_to_del.append(key)
        for item in plates_to_del:
            del plates_ever_met[item]

        frame_counter += 1
        plates_in_frame.clear()
        plates_mean_coords_in_frame.clear()

        print(frame_counter)
        # if frame_counter == 100:
        #     break
        #     # print()

    #TODO построить гистограмму ошибок (с расстоянием левенштейна для похожих слов 
    # или можно просто сверять с TARGETS и смотреть тоьлко на номера из этого списка)
    #TODO изменить тип подсчета скорости, убрать среднюю сделать мгновенную или наоборот
    #TODO заскринить одну машину и посмотреть прямо на фотках, сколько она проехала
    #TODO график разброса скоростей между соседними фреймами - отличие средней скорости как раз может отличаться от мгновенной своей усредненностью
    #TODO мерить только мгновенные скорости, брать самую близкую к эталонной и строить гистограмму ошибок

    res_file.close() 
    cv.destroyAllWindows()
