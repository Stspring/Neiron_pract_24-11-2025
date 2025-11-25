import numpy as np
from keras import models, layers

models = models.Sequential([
    layers.Dense (3, input_shape = (5,), activation="relu") 
])

weights, biases = models.layers [0].get_weights()
print(weights.shape)
print(biases.shape)
print(weights[:3,:5])
print(biases[:10])