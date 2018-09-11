#Первичное обнаружение ГРЗ в видеопотоке и распознавание её содержимого
import numpy as np
import cv2 as cv
import sys, caffe
import time



PATH_TO_MODEL = "/ssd480/talos/python/platedetection/defenitions/embeded_min.prototxt"
PATH_TO_WEIGHTS = "/ssd480/talos/python/platedetection/nets/lstm.caffemodel"
#Алфавит всех распознаваемых символов
alphabet = ["1", "А", "3", "В", "5", "С", "7", "Е", "9", "К", "4", "Х", "8", "Н", "2", "М", "О", "Р", "Т", "6", "У", "@"]


def output_plateNumber(img):
    #Выполняем нормализацию и подгонку размеров под вход сети
    image = cv.resize(img, (160, 32)) / 255.0
    net.blobs['data'].data[...] = image
    net.forward()
    set_index = []
    p = []

    for i in range(1, 11):
        blob = net.blobs["ip_sm" + str(i)]
        index = blob.data[0].argmax()
        set_index.append(index)
        p.append(blob.data[0][index])

    #print(p)

    number = ""
    for index in set_index:
        number += alphabet[index]

    return number, p



if __name__ == "__main__":
    start = time.clock()

    if len(sys.argv) > 1:
        cap = cv.VideoCapture(sys.argv[1])
    else:
        cap = cv.VideoCapture(0)
    
    plate_cascade = cv.CascadeClassifier("/ssd480/talos/python/platedetection/haar/cascade.xml")
    #Переменная введена для сокращения числа обрабатываемых кадров
    frame_counter  = 0
    caffe.set_mode_cpu()
    net = caffe.Net(PATH_TO_MODEL, PATH_TO_WEIGHTS, caffe.TEST)
    #Словарь для хранения треков
    tracks = {}
    #Файл для записи распознанных номеров
    track_results = open("tracks", "w")
    to_remove = []

    while True:
        ret, img = cap.read()
        frame_counter += 1

        if ret is False:
            print("Cannot read a stream")
            break
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(75, 15), maxSize=(150, 30))

        for (x,y,w,h) in plates:
            number, threshold = output_plateNumber(gray[y:y+h, x:x+w])
            print(number)
            if number in tracks:
                tracks[number][0] += 1
                tracks[number][1] = frame_counter
            else:
                tracks[number] = [1, frame_counter]
            cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

        cv.imshow('img',img)
        if cv.waitKey(10) == 27:
            print("quite")
            break
        

        for k, v in tracks.items():
            if frame_counter - v[1] > 25 and v[0] > 1:
                track_results.write(k + str(v[0]) + "\n")
                to_remove.append(k)
            elif frame_counter - v[1] > 25 and v[0] == 1:
                to_remove.append(k)
        
        for e in to_remove:
            tracks.pop(e)
        
        to_remove.clear()


    for k, v in tracks.items():
        if v[0] > 1:
            track_results.write(k + str(v[0]) + "\n")

    track_results.close()
    cv.destroyAllWindows()
    print(time.clock() - start)
