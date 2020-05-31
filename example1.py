#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 10:46:12 2020

@author: aguasharo
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron


iris = load_iris()
data = iris.data[:,(2,3)]
labels = iris.target

plt.figure(figsize=(13,6))
plt.scatter(data[:,2],data[:,3],c=labels)
plt.show()

y = (iris.target == 2).astype(np.int)
test_perceptron = Perceptron()
test_perceptron.fit(data, y)


y1_pred = test_perceptron.predict([[5.1,2]])
print('Prediction 1:',y1_pred)

y2_pred = test_perceptron.predict([[1.4,0.2]])
print('Prediction 2:',y2_pred)