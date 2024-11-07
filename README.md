# Neural Network Framework in Python

This is a simple neural network framework implemented in Python. It provides a modular and extensible way to build and train neural networks for various tasks.

## Features

- Supports different activation functions: ReLU, Sigmoid, Tanh, Softmax
- Supports different loss functions: Mean Squared Error (MSE), Cross-Entropy, Binary Cross-Entropy
- Includes Convolutional, Flatten, and Dense layers
- Allows for easy configuration and customization of the network architecture
- Provides a simple training loop with gradient descent

## Usage

Here's an example of how to use the `Network` class:

```python
from network import Network

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
