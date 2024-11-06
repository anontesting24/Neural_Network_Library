import numpy as np
import activations
from scipy import signal

class Layer:
    def __intit__(self):
        self.input = None
        self.output = None
        self.z = None
    def forward(self, input):
        pass
    def backward(self, output_gradient, learning_rate):
        pass

#functional but incomplete: Addition of batch processing support remaining
class Dense(Layer):
    def __init__(self, input_size, output_size, activation = None):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.zeros((output_size,1))
        self.activation=activation
    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights,self.input) + self.bias
        if self.activation==None:
            return self.output
        a = self.activation.function(self.output)
        return a
    def backward(self, output_gradient, learning_rate):
        if self.activation==None:
            continue_gradient = output_gradient
        elif issubclass(self.activation, activations.softmax):
            continue_gradient = output_gradient
        else:
            continue_gradient = output_gradient*self.activation.derivative(self.output)
        weights_gradient = np.dot(continue_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T,continue_gradient)
        self.weights -= learning_rate*weights_gradient
        self.bias -= learning_rate*continue_gradient
        return input_gradient

#functional but Incomplete: Addition of batch processing and multithreading
#Incomplete
class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, filters, activation = None):
        input_depth, input_height, input_width = input_shape
        self.activation = activation
        self.depth = filters
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (filters, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (filters, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i]+=signal.correlate2d(self.input[j], self.kernels[i,j], "valid")
        a = self.activation.function(self.output)
        return a
    def backward(self, output_gradient, learning_rate):
        if self.activation==None:
            continue_gradient = output_gradient
        elif issubclass(self.activation, activations.softmax):
            continue_gradient = output_gradient
        else:
            continue_gradient = output_gradient*self.activation.derivative(self.output)
        
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i,j]=signal.correlate2d(self.input[j], continue_gradient[i], "valid")
                input_gradient[j]+=signal.convolve2d(continue_gradient[i], self.kernels[i,j], "full")
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * continue_gradient
        return input_gradient
        

class Flatten(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)
