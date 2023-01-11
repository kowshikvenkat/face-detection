import cv2
import  numpy as num
import face_recognition

imgelon = face_recognition.load_image_file('imagesbasic/elonmusk.jpg')
imgelon = cv2.cvtColor(imgelon,cv2.cv2.COLOR_BGR2RGB)
imgetest = face_recognition.load_image_file('imagesbasic/billgates.jpg')
imgetest = cv2.cvtColor(imgetest,cv2.cv2.COLOR_BGR2RGB)
jackmaimg = face_recognition.load_image_file('imagesbasic/jackma.jpg')
faceloc =face_recognition.face_locations(imgelon)[0]
encodeelon = face_recognition.face_encodings(imgelon)[0]
cv2.rectangle(imgelon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloctest =face_recognition.face_locations(imgetest)[0]
encodeelontest = face_recognition.face_encodings(imgetest)[0]
cv2.rectangle(imgetest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeelon],encodeelontest)
facedis = face_recognition.face_distance([encodeelon],encodeelontest)
print(facedis)
cv2.putText(imgetest,f'{results}{round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
#print(faceloc)


cv2.imshow('Elon musk',imgelon)
cv2.imshow('Elon Musk', imgetest)
cv2.waitKey(0)