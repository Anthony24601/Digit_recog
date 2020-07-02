#usr/bin/python3

import tensorflow as tf
import ssl
import numpy as np

#In order to get pass the SSL BS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

#=============================================================================================#
#Exploring a data set and being able to print a picture

from tensorflow.keras.datasets import mnist
#Load data from mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#import matplotlib for use
import matplotlib.pyplot as plt

#Print out a picture
image_index = 35
print(y_train[image_index])
plt.imshow(x_train[image_index], cmap='Greys')
plt.show()

#Just making sure they are the same size
print(x_train.shape) #28 by 28
print(x_test.shape) #28 by 28
print(y_train[:image_index + 1]) #[5 0 4 1 9 2 1 3 1 4 3 5 3 6 1 7 2 8 6 9 4 0 9 1 1 2 4 3 2 7 3 8 6 9 0 5]

#=============================================================================================#
#Cleaning Data
#save image dimensions
img_rows, img_cols = 28,28 #need to figure out how to have the computer do this automatically instead of me doing it manually

#use reshape to put into format M x N x 1
x_train = x_train.reshape(x_train[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test[0], img_rows, img_cols, 1)

#Normalize data by dividing each pixel value by 255 (since RBG is 0-255)
x_train /= 255
x_test /= 255

#Convert dependent variable from integer value to binary class
from tensorflow.keras.utils import to_categorical #to_categorical converts a class vector(integer) to a binary class matrix
num_classes = 10

y_train = to_categorical(y_train,num_classes)
y_test = to_categorical(y_test,num_classes)

#=============================================================================================#
#Designing a model

#Creating a convolutional layer (takes input images)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D #all are used for input and output

model = Sequential()
#relu stands for Rectified Linear Units
#Takes either the max value or 0
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(img_rows,img_cols,1))) 

#Adding a Pooling layer
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Adding a Dropout layer
model.add(Dropout(0.25))

#Adding a flattening layer. Converts previous hidden layer to 1D layer
model.add(Flatten())

#Adding a Dense Hidden Layer
model.add(Dense(128,activation='relu'))
#Adding another Dropout layer
model.add(Dropout(0.5))
#Adding another Dense Hidden Layer
model.add(Dense(num_classes,activation='softmax')) #softmax used to classify the data into a number of predecided classes

#=============================================================================================#
#Compile and Train Model

#sparce_categorical_crossentropy is a loss function used just incase we have a integer dependent variable
#'adam' is an optimizer used for Stochastic Optimization
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Batch_size is number of training examples in one forward or backward pass
#epoch is one forward pass and one backward pass of all training examples
batch_size = 128
epochs = 10

model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))
score = model.evaluate(x_test,y_test,verbose=0)
print('Total Loss', score[0])
print('Test Accuracy', score[1])
model.save('test_model.h5')


