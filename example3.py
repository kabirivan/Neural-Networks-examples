#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:27:18 2020

@author: aguasharo
"""


from keras.models import Sequential
from keras.layers import Dense, Activation

model1 = Sequential()
model1.add(Dense(13,input_dim = 13, kernel_initializer='normal', activation = 'relu'))
model1.add(Dense(6, kernel_initializer='normal', activation = 'relu'))
model1.add(Dense(4, kernel_initializer='normal', activation = 'relu'))
model1.add(Dense(1, kernel_initializer='normal'))

model1.compile(loss='mse', optimizer = 'adam', metrics=['mean_absolute_percentage_error'])

#print(model.summary())

#from keras.utils import plot_model
#plot_model(model1, to_file = 'firstModel.png', show_shapes=True)
