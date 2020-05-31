#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:27:18 2020

@author: aguasharo
"""


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split = 0.2, seed = 10)



model1 = Sequential()
model1.add(Dense(13,input_dim = 13, kernel_initializer='normal', activation = 'relu'))
model1.add(Dense(6, kernel_initializer='normal', activation = 'relu'))
model1.add(Dense(6, kernel_initializer='normal', activation = 'relu'))
model1.add(Dense(1, kernel_initializer='normal'))

model1.compile(loss='mse', optimizer = 'adam', metrics=['mean_absolute_percentage_error'])

#print(model.summary())
#from keras.utils import plot_model
#plot_model(model1, to_file = 'firstModel.png', show_shapes=True)

# Training and evaluation 

x_val =  x_train[300:,]
y_val = y_train[300:,]
#model1.fit(x_train,y_train, batch_size=32, epochs=100, validation_data=(x_val, y_val))


history1 = model1.fit(x_train,y_train, batch_size=64, epochs = 300, validation_data=(x_val, y_val), verbose = 0)
import matplotlib.pyplot as plt
# plt.plot(history1.history['mean_absolute_percentage_error'])
# plt.plot(history1.history['val_mean_absolute_percentage_error'])

plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])


result = model1.evaluate(x_test, y_test)
print(result)
