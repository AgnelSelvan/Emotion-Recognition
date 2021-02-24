from time import sleep
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np 
import cv2 as cv2
import sys

# sys.path.append('/home/agnel/Agnel/ML/Emotion and Gender Detector/')
from trainModel import *

classifier = load_model('./TrainedModel/4(57)/Emotion_detector.h5')

face_classfier = cv2.CascadeClassifier('./HaarCascade/haarcascade_frontalface_default.xml')

def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classfier.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x-50, y-50), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]

    try:
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    except:
        return (x, y, w, h), np.zeros((48, 48), np.uint8), img
    
    return (x, y, w, h), roi_gray, img

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    rect, face, image = face_detector(frame)
    if np.sum([face]) != 0.0:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]
        label_position = (rect[0] + int((rect[1]/2)), rect[2] + 25)
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
    
    else:
        cv2.putText(image, "No Face found", (20, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    
    cv2.imshow('All', image)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
