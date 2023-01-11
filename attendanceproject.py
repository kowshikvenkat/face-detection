import cv2
import  numpy as num
import face_recognition
import os
from datetime import datetime

path = 'imagesattendance'
images = []
classnames = []
mylist = os.listdir(path)
for cls in mylist:
    curimg = cv2.imread(f'{path}/{cls}')
    images.append(curimg)
    classnames.append(os.path.splitext(cls)[0])
print(classnames)

def findencodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist
    print('encoding complete')

def markattendance(name):
    with open('attendance.csv','r+') as f:
        mydatalist = f.readlines()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

encodingknown = findencodings(images)
cap = cv2.VideoCapture(0)

while True:
    success , img = cap.read()
    imgsmall = cv2.resize(img,(0,0),None,0.25,0.25)

    imgsmall = cv2.cvtColor(imgsmall, cv2.cv2.COLOR_BGR2RGB)

    facecurframe = face_recognition.face_locations(imgsmall)
    encode = face_recognition.face_encodings(imgsmall,facecurframe)

    for encodeface,faceloc in zip(encode,facecurframe):
        matches = face_recognition.compare_faces(encodingknown,encodeface)
        facedis = face_recognition.face_distance(encodingknown,encodeface)
        #print(facedis)
        matchindex = num.argmin(facedis)

        if matches[matchindex]:
            name = classnames[matchindex].upper()
            #print(name)
            y1,x2,y2,x1 = faceloc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4 , x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattendance(name)


        cv2.imshow('Webcam',img)
        cv2.waitKey(1)

