import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import os
import tqdm
import cv2
import math
import glob
from keras_preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16


base_model = VGG16(weights='imagenet', include_top=False)
model = load_model("model.h5")
data = []

cap = cv2.VideoCapture(0)
success,frame = cap.read()


if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    im = cv2.resize(frame, (224,224))
    im = im.reshape((224,224,3))
    im = im / 255
    data.append(im)
    cv2.imshow('Input', frame)



    c = cv2.waitKey(1)
    if c == 27:
        data = np.asarray(data)
        data = base_model.predict(data)
        data = data.reshape(len(data),7*7*512)
        a = model.predict_classes(data)
        print(a)
        break

cap.release()
cv2.destroyAllWindows()
