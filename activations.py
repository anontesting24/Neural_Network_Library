#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

class sigmoid:
    @staticmethod
    def function(z):
        return 1/(1 + np.exp(-z))
    @staticmethod
    def derivative(z):
        s = 1/(1 + np.exp(-z))
        return s * (1-s)
class tanh:
    @staticmethod
    def function(z):
        return np.tanh(z)
    @staticmethod
    def derivative(z):
        return 1 - np.tanh(z) ** 2
class relu:
    @staticmethod
    def function(z):
        return np.maximum(0,z)
    
    @staticmethod
    def derivative(z):
        return np.where(z > 0, 1, 0)
        
class softmax:
    @staticmethod
    def function(z):
        e_pa=np.exp(z)
        ans=e_pa/np.sum(e_pa)
        return ans
    @staticmethod
    def derivative(z):
        '''
        n = np.size(z)
        return np.identity(n) - z.T) * z
        '''
        print("softmax not usable in middle layers")
        pass

