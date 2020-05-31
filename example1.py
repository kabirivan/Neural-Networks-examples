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
plt.scatter(data[:,0],data[:,1],c=labels)
plt.show()
adfsa