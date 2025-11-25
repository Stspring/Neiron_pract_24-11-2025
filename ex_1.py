import numpy as np
from keras import models, layers

model = models.Sequential([
    layers.Dense (3, input_shape = (5,), activation="relu") 
])

weights = np.full((5,3),0.5)
biases = np.zeros(3)
model.layers[0].set_weights([weights, biases]) 
x = np.array([[1,2,3,4,5],[0,1,0,1,0]])
y = model.predict(x)
print("Enters: ", x)
print("Outputs: ", y)
