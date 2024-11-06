#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

class mse:
    @staticmethod
    def function(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))
    @staticmethod
    def derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)

class binary_cross_E:
    @staticmethod
    def function(y_true, y_pred):
        return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
    @staticmethod
    def derivative(y_true, y_pred):
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

class cross_E:
    @staticmethod
    def function(y_true, y_pred):
        return -np.sum(y_true*np.log(y_pred + 10**-100))
    @staticmethod
    def derivative(y_true, y_pred):
        #assuming optimization softmax
        return (y_pred - y_true)
    

