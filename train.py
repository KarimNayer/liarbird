import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import glob

videos = pd.read_csv('videos_frames.csv')
print(videos.shape[0])
data = []
for i in tqdm(range(videos.shape[0])):
    # loading the image and keeping the target size as (224,224,3)
    img = image.load_img(videos['image'][i], target_size=(224, 224, 3))
    # converting it to array
    img = image.img_to_array(img)
    # normalizing the pixel value
    img = img / 255
    # appending the image to the train_image list
    data.append(img)
    print(img.shape)

# converting the list to numpy array
X = np.array(data)

y = videos['class']
Y = []
for i in y:
    if i == "lie":
        Y.append(0)
    if i == "truth":
        Y.append(1)
# creating the training and validation set
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2, stratify = Y)
# creating the base model of pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)
print(X_train.shape)
X_train = base_model.predict(X_train)
X_test = base_model.predict(X_test)

X_train = X_train.reshape(len(X_train), 7*7*512)
X_test = X_test.reshape(len(X_test), 7*7*512)

# reshaping the training as well as validation frames in single dimension
max = X_train.max()
X_train = X_train/max
X_test = X_test/max

print(X_train.shape)
print(X_test.shape)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(7*7*512,)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='sigmoid'))
#mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')
model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=60)
a = model.evaluate(X_test,y_test)
model.save("model.h5")
