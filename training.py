import os
import cv2
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import svm
import tqdm
from keras.preprocessing import image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from keras.applications.resnet50 import ResNet50
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import SGD
# Creating the target variable

train_x,train_y=[],[]
training_data=[]
for img in os.listdir("train/Level_1/"):
    # print(img)
    # img=cv2.imread("train/Level_1/"+str(img))
    # img=cv2.resize(img,(32,32))
    # print(img.shape)
    # cv2.imshow("im",img)
    # cv2.waitKey(0)
    train_x.append("train/Level_1/"+str(img))
    if(img.split("_")[0]=='colon'):
        class_num=10
    elif(img.split("_")[0]=='dot'):
        class_num=11
    else:
        class_num=int(img.split("_")[0])
        # train_y.append(img.split("_")[0])

    img_array = cv2.imread("train/Level_1/"+str(img) ,cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (28, 28)) 
    training_data.append([new_array, class_num])
print(training_data)




X = []
y = []
for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X, dtype="float") / 255.0

le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)
print(y,'y')


aug = ImageDataGenerator()

aug = ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")
# print(X[0].reshape(-1, 28, 28, 1))

X = np.array(X).reshape(-1, 28, 28, 1)
# y=np.array(y)


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())  

model.add(Dense(128))
model.add(Dropout(0.25))

model.add(Dense(12))
model.add(Activation('softmax'))
opt = SGD(lr=1e-1, momentum=0.9, decay=1e-1 / 30)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x=aug.flow(X, y, batch_size=32),steps_per_epoch=len(X) // 32, epochs=20)
model.save("digit.h5")
# test_x=cv2.imread("train/Level_one/2/2_1.jpg")
# # test_x=cv2.imread("test/1/test_1.jpg")
# test_x=cv2.resize(test_x,(28,28))
# cv2.imshow("imgg",test_x)
# cv2.waitKey(0)
# # test_x=np.array(test_x)

# test_x = np.array(test_x, dtype="float") / 255.0
# test_x = np.array(test_x).reshape(-1, 28, 28, 1)

# print(model.predict(test_x))
# train_image=[]
# for i in train_x:
#     img = image.load_img(str(i), target_size=(28,28,1), grayscale=True)
#     img = image.img_to_array(img)
#     img = img/255
#     train_image.append(img)
# X = np.array(train_image)
# print(X)
# train_y=to_categorical(train_y)
# train_y=np.array(train_y)
# print(train_y),'ty'
# print(len(train_image),len(train_y))
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))
# # # Compile the model
# model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
# # Training the model
# model.fit(train_image, train_y, epochs=1)
# print(train_x,train_y)
# train_new_x=[]
# for img in train_x:
#     img=cv2.imread(img)
#     img=cv2.resize(img,(32,32))
#     # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     im=np.array(img)
#     print(im.shape)
#     train_new_x.append(im)
# train_new_x=np.array(train_new_x)
# print(train_new_x)
# train_y=np.array(train_y)
# clf = svm.SVC()
# clf.fit(train_image,train_y)


    