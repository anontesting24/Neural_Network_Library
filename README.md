# Neural Network Framework in Python

This is a simple neural network framework implemented in Python. It provides a modular and extensible way to build and train neural networks for various tasks.

Shout out to @omar_aflak/The Independent Code for creating impeccable content to learn about neural networks

## Features

- Supports different activation functions: ReLU, Sigmoid, Tanh, Softmax
- Supports different loss functions: Mean Squared Error (MSE), Cross-Entropy, Binary Cross-Entropy
- Includes Convolutional, Flatten, and Dense layers
- Allows for easy configuration and customization of the network architecture
- Provides a simple training loop with gradient descent

'''
### Activation and Loss Functions

The framework provides a dictionary of available activation and loss functions. To use a specific function, you can pass the corresponding key as a string when adding layers or setting the loss function.

Available activation functions:
* 'relu': activations.relu
* 'sigmoid': activations.sigmoid 
* 'tanh': activations.tanh
* 'softmax': activations.softmax

Available loss functions:
* 'mse': losses.mse
* 'cross_entropy': losses.cross_E
* 'binary_cross_entropy': losses.binary_cross_E

## File Structure

* `network.py`: Contains the `Network` class and its methods.
* `layers.py`: Defines the different layer types (Convolutional, Flatten, Dense).
* `activations.py`: Implements the activation functions (ReLU, Sigmoid, Tanh, Softmax).
* `losses.py`: Implements the loss functions (MSE, Cross-Entropy, Binary Cross-Entropy).

## Dependencies

* `numpy`
* `scipy` 
* `matplotlib`

## Future Improvements

* Add support for more layer types (e.g., Pooling, Dropout, BatchNormalization)
* Implement more advanced optimization algorithms (e.g., Adam, RMSProp)
* Add support for GPU acceleration


## Usage

Here's an example of how to use the `Network` class:

```python
from neuralnet import Network

# Create a new network
net = Network()

# Set the loss function
net.loss('cross_entropy')

# Add layers to the network
net.add_Convolutional('relu', kernel_size=3, filters=32, input_shape=(1, 28, 28))
net.add_Flatten()
net.add_Dense('relu', output_size=64)
net.add_Dense('softmax', output_size=10)

# Load training data
net.load_training_data(x_train, y_train)

# Train the network
net.train(eps=100, lr=0.1)

# Make a prediction
output = net.predict(input_data)
