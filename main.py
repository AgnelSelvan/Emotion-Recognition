# import pickle
# import numpy as np
# import cv2

# face_cascade = cv2.CascadeClassifier('/home/agnel/Agnel/files/HaarCascade/haarcascade_profileface.xml')
# eye_cascade = cv2.CascadeClassifier('/home/agnel/Agnel/files/HaarCascade/haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier('/home/agnel/Agnel/files/HaarCascade/haarcascade_smile.xml')
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read("trainner.yml")

# labels = {}
# with open('label.pickle', 'rb') as f:
#     og_labels = pickle.load(f)
#     labels = {v:k for k, v in og_labels.items()}

# cap = cv2.VideoCapture(0)
# i = 0
# stroke = 2
# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
#     for (x, y, w, h) in faces:
#         #print(x, y, w, h)
#         img_item = str(i)+".png"
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]
        
#         id_, conf = recognizer.predict(roi_gray)
#         #print(labels[id_])
#         cv2.putText(frame, labels[id_], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), stroke)
                
#         color= (0, 0, 255) #BGR
#         stroke = 2
#         width = x + w
#         height = y + h
#         cv2.rectangle(frame, (x - 10, y), (width + 10, height + 20), color, stroke)
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         for (ex, ey, ew, eh) in eyes:
#             cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0))
#         # smile = smile_cascade.detectMultiScale(roi_gray)
#         # for (sx, sy, sw, sh) in smile:
#         #     cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 0))
    
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) == 13:
#         break
    
# cap.release()
# cv2.destroyAllWindows()

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import ELU
import os

num_classes = 6
img_rows, img_cols = 48, 48
batch_size = 16

train_data_dir = './images/train'
validation_data_dir = './images/validation'

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=30,
        shear_range=0.3,
        zoom_range=0.3,
        width_shift_range=0.4,
        height_shift_range=0.4,
        horizontal_flip=True,
        fill_mode='nearest'    
    )

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        color_mode='grayscale',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)


model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal', input_shape=(img_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer='he_normal', input_shape=(img_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# model.add(Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal'))
# model.add(Activation('elu'))
# model.add(BatchNormalization())
# model.add(Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal'))
# model.add(Activation('elu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal'))
# model.add(Activation('elu'))
# model.add(BatchNormalization())
# model.add(Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal'))
# model.add(Activation('elu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes, kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())

validatin_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False    
)

class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())
print(class_labels)

# class_labels = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
# print(class_labels)

from keras.preprocessing import image
from keras.optimizers import RMSprop, SGD, Adam
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import re

# BLACK = [0, 0, 0]
# def draw_test(name, pred, im, true_label):
#     expanded_image = cv2.copyMakeBorder(im, 160, 0, 0, 300, cv2.BORDER_CONSTANT, value=BLACK)
#     cv2.putText(expanded_image, "Predicted: "+pred, (20, 60), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2 )
#     cv2.putText(expanded_image, "True: "+true_label, (20, 120), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2 )
#     cv2.imshow(name, expanded_image)

# def getRandomImage(path, img_width, img_height):
#     folders = list(filter(lambda x:os.path.isdir(os.path.join(path, x)), os.listdir(path)))
#     random_directory = np.random.randint(0, len(folders))
#     path_class = folders[random_directory]
#     file_path = path + path_class
#     file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
#     random_file_index = np.random.randint(0, len(file_names))
#     image_name = file_names[random_file_index]
#     final_path = file_path+"/"+image_name
#     return image.load_img(final_path, target_size=(img_width, img_height), grayscale=True), final_path, path_class

# img_width, img_height = 48, 48

# model.compile(loss='categorical_crossentropy',
#              optimizer=RMSprop(lr=0.001),
#               metrics=['accuracy']
#              )

# files = []
# predictions = []
# true_labels = []

# for i in range(0, 10):
#     path = './images/validation/'
#     img, final_path, true_label = getRandomImage(path, img_width, img_height)
#     files.append(final_path)
#     true_labels.append(true_label)
#     x = image.img_to_array(img)
#     x = x * 1. / 255
#     x = np.expand_dims(x, axis = 0)
#     images = np.vstack([x])
#     classes = model.predict_classes(images, batch_size=10)
#     predictions.append(classes)

# for i in range(0, len(files)):
#     image = cv2.imread((files[i]))
#     image = cv2.resize(image, None, fx = 3, fy=3, interpolation=cv2.INTER_CUBIC)
#     draw_test("Prediction", class_labels[predictions[i][0]], image, true_labels[i])
#     cv2.waitKey(0)

# cv2.destroyAllWindows()

from keras.preprocessing.image import img_to_array

face_classifier = cv2.CascadeClassifier('/home/agnel/Projects/Python/PersonalFaceAndEmotionRecognition-master (1)/HaarCascade/haarcascade_frontalface_default.xml')

def face_detector(img):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img
    
    all_faces = []
    rects = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        all_faces.append(roi_gray)
        rects.append((x, y, w, h))
    return rects, all_faces, img

img = cv2.imread("28745.jpg")
rects, faces, image = face_detector(img)

i = 0
for face in faces:
    roi = face.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis = 0)
    
    preds = classifier.predict(roi)[0]
    label = class_labels[preds.argmax()]
    
    label_position = (rects[i][0] + int((rects[i][1]/2)), abs(rects[i][2] - 10))
    i += 1
    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    
cv2.imshow("Emotion Detection:", image)
cv2.waitKey(0)

cv2.destroyAllWindows()

