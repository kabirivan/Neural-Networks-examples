#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:01:11 2020

@author: aguasharo
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier


iris = load_iris()

test_MLPC = MLPClassifier()

test_MLPC.fit(iris.data, iris.target) 

y1 = test_MLPC.predict([[4.7, 3.5, 1.3, 0.2]])
y2 = test_MLPC.predict([[3.7, 2.1, 4.3, 1.8]])

print('0 -> Iris Setosa\n1 -> Iris Versicolor \n2 -> Iris Virginica \n\n')
print('1. La prediccion es {y1}')
print('1. La prediccion es {y2}')