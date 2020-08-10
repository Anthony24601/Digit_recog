#usr/bin/python3
#Second Version

import tensorflow as tf
import ssl
import numpy as np
import pandas as pd
import tensorflow.keras.wrappers.scikit_learn as scikit_learn

#In order to get pass the SSL bs on my computer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
from tensorflow.keras.wrappers.scikit_learn import train_test_split





from tensorflow.keras.datasets import mnist
#Load data from mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#import matplotlib for use
from matplotlib import pyplot as plt

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

#use .reshape to put into format M x N x 1
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

#Normalize data by dividing each pixel value by 255 (since RBG is 0-255)
x_train = x_train / 255 
x_test = x_test / 255 

#Convert dependent variable from integer value to binary class
from tensorflow.keras.utils import to_categorical #to_categorical converts a class vector(integer) to a binary class matrix
num_classes = 10

y_train = to_categorical(y_train,num_classes)
y_test = to_categorical(y_test,num_classes)

#=============================================================================================#
#Designing a model
#CNN goes In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

#Creating a convolutional layer (takes input images)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D #all are used for input and output

model = Sequential()
#relu stands for Rectified Linear Units
#Takes either the max value or 0
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(img_rows,img_cols,1),padding='Same'))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='Same')) 

#Adding a Pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

#Adding a Dropout layer
model.add(Dropout(0.25))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='Same'))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='Same'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

#Adding a flattening layer. Converts previous hidden layer to 1D layer
model.add(Flatten())

#Adding a Dense Hidden Layer
model.add(Dense(256,activation='relu'))
#Adding another Dropout layer
model.add(Dropout(0.5))
#This is a output layer that outputs the distribution of probability for each class or case
model.add(Dense(num_classes,activation='softmax'))

#=============================================================================================#
#Compile and Train Model

#RMSprop (with default values), it is a very effective optimizer. The RMSProp update adjusts the Adagrad method in a very simple way in an attempt to reduce its aggressive, monotonically decreasing learning rate.
#Faster than Stochastic Gradient Descent
from tensorflow.keras.optimizers import RMSprop
optimizer = RMSprop(learning_rate=0.001, rho=0.9,epsilon=1e-08,decay=0.0)

#categorical_crossentropy is a loss function used since we have an output of a vector-based dependent variable
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

#In order to make the optimizer converge faster and closest to the global minimum of the loss function
#We will use a Anealing method for the learning rate
#We will reduce the learning rate by half if accuracy hasn't improved after 3 epochs
from tensorflow.keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,verbose=1,min_lr=0.00001)

batch_size = 86
epochs = 2 #change to 30 to be more accurate

#Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(featurewise_center=False,
                            samplewise_center=False,
                            featurewise_std_normalization=False,
                            samplewise_std_normalization=False,
                            zca_whitening=False,
                            rotation_range=10,
                            zoom_range=0.1,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            vertical_flip=False,
                            horizontal_flip=False)

datagen.fit(x_train)

history = model.fit(x=datagen.flow(x_train,y_train,batch_size=batch_size),
                    epochs=epochs,verbose=2,validation_data=(x_test,y_test), #change this to x_val and y_val in future when make validation set
                    steps_per_epoch=x_train.shape[0]//batch_size,callbacks = [learning_rate_reduction])

score = model.evaluate(x_test,y_test,verbose=0)
print('Total Loss', score[0])
print('Test Accuracy', score[1])
model.save('test_model.h5')

#=============================================================================================#
#Testing
import imageio
import numpy as np

im = imageio.imread("https://i.imgur.com/a3Rql9C.png")

#Converting the RGB Values to grayscale
gray = np.dot(im[...,:3], [0.299,0.587,0.114])
plt.imshow(gray, cmap= plt.get_cmap('gray'))
plt.show()

#reshape image
gray = gray.reshape(1, img_rows, img_cols, 1)
#Normalize values
gray /=255

#load Model
from tensorflow.keras.models import load_model
model = load_model("test_model.h5")

#predict digit
prediction = model.predict(gray)
print(prediction.argmax())
