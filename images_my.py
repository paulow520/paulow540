import numpy as np
import cv2 
import face_recognition

# create new environment conda create -n env_dlib python=3.6
# activate enviroment conda activate env_dlib
# install dlib conda install -c conda-forge dlib

imgpaul = face_recognition.load_image_file("allimage/elontrain.jfif")
# imgpaul = cv2.cvtColor(imgpaul, cv2.COLOR_BAYER_BG2RGB)


imgpaultest = face_recognition.load_image_file("allimage/paul.png")
# imgpaultest = cv2.cvtColor(imgpaul, cv2.COLOR_BAYER_BG2RGB)


faceloct = face_recognition.face_locations(imgpaul)[0]
encodepaul = face_recognition.face_encodings(imgpaul)[0]
# print(faceloct)
cv2.rectangle(imgpaul,(faceloct[3],faceloct[0]),(faceloct[1],faceloct[2]),(255,0,255),2)

faceloctest = face_recognition.face_locations(imgpaultest)[0]
encodepaultest = face_recognition.face_encodings(imgpaultest)[0]
cv2.rectangle(imgpaultest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

myresult = face_recognition.compare_faces([encodepaul],encodepaultest)

facedis = face_recognition.face_distance([encodepaul],encodepaultest)

cv2.putText(imgpaultest, f"{myresult} {round(facedis[0],2)}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

print(myresult, facedis)

cv2.imshow("paul image", imgpaul)
cv2.imshow("paultext image", imgpaultest)
cv2.waitKey(0)
