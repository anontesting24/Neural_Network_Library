#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from neuralnet import Network

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

firstmodel=Network()
firstmodel.add_Dense('tanh',3,2)
firstmodel.add_Dense('tanh',1)
firstmodel.load_training_data(X,Y)
firstmodel.loss('mse')
firstmodel.train(1000)

points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = [[x], [y]]
        z = firstmodel.predict(z)
        points.append([x, y, z[0,0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2])
plt.show()

