#Первичное обнаружение ГРЗ в видеопотоке и распознавание её содержимого
import numpy as np
import cv2 as cv
import sys, caffe
import time
from os.path import join


PATH_TO_MODEL = "/home/grigorii/ssd480/talos/python/platedetection/zoo/twins/embeded.prototxt"
PATH_TO_WEIGHTS = "/home/grigorii/ssd480/talos/python/platedetection/zoo/twins/weights.caffemodel"
PATH_TO_CASCADE = "/home/grigorii/ssd480/talos/python/platedetection/haar/cascade_inversed_plates.xml"
PATH_TO_VIDEO = '/home/grigorii/Desktop/primary_search/video-1.mp4'
PATH_TO_SAVE_FRAMES = '/home/grigorii/Desktop/homography/test_frames'
# PATH_TO_TRACKS = '/home/adel/Documents/Python/test/meta/1'

alphabet = ['1','A','0','3','B','5','C','7','E','9','K','4','X','8','H','2','M','O','P','T','6','Y','@']
minSize_ = (50,10)
maxSize_ = (200,40)

#TODO как вытащить время из кадров

def print_(s):
    print('-'*80)
    print(s)
    print('-'*80)


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


if __name__ == "__main__":
    counter  = 0
    start = time.clock()

    # if len(sys.argv) > 1:
    #     cap = cv.VideoCapture(sys.argv[1])
    # else:
    #     cap = cv.VideoCapture(0)
    cap = cv.VideoCapture(PATH_TO_VIDEO)
    
    plate_cascade = cv.CascadeClassifier(PATH_TO_CASCADE)
    caffe.set_mode_cpu()
    net = caffe.Net(PATH_TO_MODEL, PATH_TO_WEIGHTS, caffe.TEST)
    # tracks = target_list(PATH_TO_TRACKS)
    # frame_numbers = open(PATH_TO_INCORRECTLY_CLASSIFIED_IMGS+'frame_numbers','w')
    # all_numbers = open(PATH_TO_INCORRECTLY_CLASSIFIED_IMGS+'all_numbers','w')

    while True:
        ret, img = cap.read()
        if ret is False:
            print("Cannot read a stream")
            break
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=minSize_, maxSize=maxSize_)

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
            try:
            	cv.rectangle(gray,(x,y-d),(x+w,y+h+d),(0,255,0),1)
            except:
            	cv.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),1)
            cv.imwrite(join(PATH_TO_SAVE_FRAMES, str(counter) + '.png'), img)
            counter += 1
            print(counter)

        # cv.imshow('img',img)
        # if cv.waitKey(70) == 27:
        #     print("quite")
        #     break


    cv.destroyAllWindows()
    print(time.clock() - start)
