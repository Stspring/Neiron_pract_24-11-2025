import numpy as np
from keras import models, layers


kernel_initializer = 'he_normal'
bias_initializer = 'ones'

model = models.Sequential([
    layers.Dense(64, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, input_shape=(100,))
])

w, b = model.layers[0].get_weights()

print(f'Kernel initializer: {kernel_initializer}')
print(f'Weights - mean: {np.mean(w)}, variance: {np.var(w)}')

print(f'Bias initializer: {bias_initializer}')
print(f'Biases - mean: {np.mean(b)}, variance: {np.var(b)}')
