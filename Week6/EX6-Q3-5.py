# -*- coding: utf-8 -*-

from traffic_signs import load_data 


import numpy as np
from sklearn.model_selection import train_test_split



from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
#from keras.utils import to_categorical



#----------------------------  Q3 -----------------------------------------

X, y = load_data("GTSRB_subset_2")

X=X-np.min(X)
X=X/np.max(X)


y1=np.int32(y==0)


y2=np.transpose(np.array([y , y1]))
X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2)


#----------------------------  Q4 -----------------------------------------

model = Sequential()

N = 32 # Number of feature maps
w, h = 5, 5 # Conv. window size

model.add(Conv2D(N, (w, h),
input_shape=(64, 64, 3),
activation = 'relu',
padding = 'same'))

model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Conv2D(N, (w, h),
activation = 'relu',
padding = 'same'))

model.add(MaxPooling2D((4,4)))

model.add(Flatten())

model.add(Dense(100, activation = 'sigmoid')) 

model.add(Dense(2, activation = 'sigmoid')) 


model.summary()


#----------------------------  Q5 -----------------------------------------


model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=20, batch_size=32)



















