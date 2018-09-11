#Первичное обнаружение ГРЗ в видеопотоке и распознавание её содержимого
import numpy as np
import cv2 as cv
import sys, caffe
import time


PATH_TO_MODEL = "/ssd480/talos/python/platedetection/defenitions/embeded.prototxt"
PATH_TO_WEIGHTS = "/ssd480/talos/python/platedetection/nets/lstm.caffemodel"
PATH_TO_INCORRECTLY_CLASSIFIED_IMGS = "/home/adel/neg/"
PATH_TO_CASCADE = "/ssd480/talos/python/platedetection/haar/cascade_inversed_plates.xml"
PATH_TO_TRACKS = '/home/adel/Documents/Python/test/meta/1'

#Алфавит всех распознаваемых символов
alphabet = ['1','A','3','B','5','C','7','E','9','K','4','X','8','H','2','M','O','P','T','6','Y','@']
minSize_ = (50,10)
maxSize_ = (200,40)


def output_plateNumber(img):
    #Выполняем нормализацию и подгонку размеров под вход сети
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
    #print("    Recognized number is ", number)
    return number

def target_list(path_to_file):
    with open(path_to_file) as f:
        content = f.readlines()
        content = [x[:-1] for x in content]
        content = [x+'@'*(10-len(x)) for x in content]
        return content

def check_number(number):
	insert = []
	for i, lett in enumerate(number[1:4]):
        #TODO additional checks
		if lett == 'O':
			insert.append('0')
		else:
			insert.append(lett)
	new_number = number[:1] + ''.join(insert) + number[len(insert)+1:]
	return new_number


if __name__ == "__main__":
    start = time.clock()

    if len(sys.argv) > 1:
        cap = cv.VideoCapture(sys.argv[1])
    else:
        cap = cv.VideoCapture(0)
    
    plate_cascade = cv.CascadeClassifier(PATH_TO_CASCADE)
    #Переменная введена для сокращения числа обрабатываемых кадров
    frame_counter  = 0
    caffe.set_mode_cpu()
    net = caffe.Net(PATH_TO_MODEL, PATH_TO_WEIGHTS, caffe.TEST)
    tracks = target_list(PATH_TO_TRACKS)
    frame_numbers = open(PATH_TO_INCORRECTLY_CLASSIFIED_IMGS+'frame_numbers','w')
    all_numbers = open(PATH_TO_INCORRECTLY_CLASSIFIED_IMGS+'all_numbers','w')

    while True:
        ret, img = cap.read()
        frame_counter += 1

        if ret is False:
            print("Cannot read a stream")
            break
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=minSize_, maxSize=maxSize_)

        for (x,y,w,h) in plates:
            d = 0
            try:
            	number = output_plateNumber(gray[y-d:y+h+d, x:x+w])
            except:
            	number = output_plateNumber(gray[y:y+h, x:x+w])
            number = check_number(number)
            all_numbers.write(str(number)+'\n')
            print(number)
            if number not in tracks:
                try:
                	cv.rectangle(img,(x,y-d),(x+w,y+h+d),(0,255,0),1)
                except:
                	cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
                cv.imwrite(PATH_TO_INCORRECTLY_CLASSIFIED_IMGS+str(frame_counter)+'.png',img)
                frame_numbers.write('frame_'+str(frame_counter)+' '+str(number)+'\n')

        # cv.imshow('img',img)
        # if cv.waitKey(70) == 27:
        #     print("quite")
        #     break


    cv.destroyAllWindows()
    print(time.clock() - start)
