import os
import cv2
import numpy as np

location = 'blank'
save_dir = 'datasets/blank/' 

index = 1
print(len(os.listdir(location)))
for filename in os.listdir(location):
    filepath = os.path.join(location, filename)
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    faces = sorted(faces, key = lambda x: -x[2])
    for (x,y,w,h) in faces:
        if w < 500 or h < 500:
            print("small")
            break
        face_img = img[y-30:y+h+30, x-30:x+w+30]
        try:
            face_img = cv2.resize(face_img, (224, 224))
        except:
            print("except")
            break
        save_path = save_dir+str(index)+'.jpg'
        cv2.imwrite(save_path, face_img)
        index += 1
        print(save_path, len(faces))
        break