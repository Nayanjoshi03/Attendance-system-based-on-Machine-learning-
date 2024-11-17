from sklearn.neighbors import KNeighborsClassifier
import pickle
import cv2
import numpy as np
import time as t
from datetime import datetime


# Loading the data from pickle file 
with open('data/names.pkl', 'rb') as f:
    LABELS=pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)

FACES=FACES/255
# Model training 
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES,LABELS)


video=cv2.VideoCapture(0)
face_detect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
flag=1
prediction=""
while(True):
    ret, frame = video.read()
    if not ret:
        print('Canot access camara')
        exit()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # conveting the colour frame into grayscale images
    faces = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        crop_image=frame[x:x+h,y:y+w]# Extracting the area of intrest 
        resized_img=cv2.resize(crop_image,(50,50)).flatten().reshape(1,-1)
        prediction=knn.predict(resized_img)
        cv2.putText(frame,str(prediction[0]),(x,y-15),cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h),(200,200,200),2)
    cv2.imshow('video',frame)
    if flag==1:
        current_time = datetime.now()
        time = current_time.strftime("%Y-%m-%d %H:%M:%S")  
        with open("attandence.csv",'a') as f:
            f.write(time +","+prediction[0]+"\n")
    if (flag==100):
        print("Your data is saved")
        flag=1
    flag+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()

# Get the current system time

    

                                        
    


