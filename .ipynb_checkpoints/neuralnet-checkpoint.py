#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import layers
import activations
import losses

class Network:
    def __init__(self):
        self.num_layers=0
        self.x_train=None
        self.y_train=None
        self.prev_output_shape=None
        self.active_dict={'relu':activations.relu, 'sigmoid':activations.sigmoid, 'tanh':activations.tanh, 'softmax':activations.softmax}
        self.loss_dict={'mse':losses.mse, 'cross_entropy':losses.cross_E, 'binary_cross_entropy':losses.binary_cross_E}
        self.lossclass=None
        self.summarize=[]
        self.summaryloss=None
        self.net=[]
        
    def summary(self):
        print("loss func is ",self.summaryloss)
        for i in self.summarize:
            print(i,"\n")
        
    
    def loss(self, lossf):
        self.summaryloss=lossf
        self.lossclass=self.loss_dict[lossf]
        
    
    def load_training_data(self, input_data, output_data):
        self.x_train = input_data
        self.y_train = output_data
        
    def add_Convolutional(self, activationf, kernel_size, filters, input_shape=None):
        if(self.num_layers==0 and input_shape==None):
            raise Exception("kindly add input shape for first layer")
            return
        if self.num_layers>0:
            input_shape=self.prev_output_shape
        
        
        input_depth, input_height, input_width = input_shape
        self.num_layers+=1
        self.net.append(layers.Convolutional(input_shape, kernel_size, filters, self.active_dict[activationf]))
        self.summarize.append('Convolutional')
        self.prev_output_shape = (filters, input_height - kernel_size + 1, input_width - kernel_size + 1)

    def add_Flatten(self, output_shape=None, input_shape=None):
        if(self.num_layers==0 and input_shape==None):
            raise Exception("kindly add input shape for first layer")
            return
        if self.num_layers>0:
            input_shape=self.prev_output_shape
        if output_shape==None:
            output_shape=1
            for i in input_shape:
                output_shape*=i
            output_shape=(output_shape,1)
        
        self.num_layers+=1
        self.net.append(layers.Flatten(input_shape, output_shape))
        self.summarize.append('Flatten')
        self.prev_output_shape=output_shape[0]
        
        
    def add_Dense(self, activationf, output_size, input_size=None):
        if(self.num_layers==0 and input_size==None):
            raise Exception("kindly add input shape for first layer")
            return
        if self.num_layers>0:
            input_size=self.prev_output_shape
        self.num_layers+=1
        self.net.append(layers.Dense(input_size, output_size, self.active_dict[activationf]))
        self.summarize.append('Dense')
        self.prev_output_shape=output_size
        
    def predict(self, input):
        output = input
        for layer in self.net:
            output = layer.forward(output)
        return output

    def train(self, eps=100, lr=0.1):
        epochs = eps
        learning_rate = lr
        
        for e in range(epochs):
            error = 0
            for x, y in zip(self.x_train, self.y_train):
                # forward
                output = self.predict(x)
        
                # error
                error += self.lossclass.function(y, output)
        
                # backward
                grad = self.lossclass.derivative(y, output)
                for layer in reversed(self.net):
                    grad = layer.backward(grad, learning_rate)
        
            error /= len(self.x_train)
            print(f"{e + 1}/{epochs}, error={error}")
        

        

