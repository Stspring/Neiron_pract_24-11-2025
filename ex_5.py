import numpy as np
from keras import models, layers

model = models.Sequential([
    layers.Dense (1, input_shape = (5,), activation="relu")     # Создаем слой
])

weight = input('Введите значение веса ')
weights = np.full((5,1), weight)     # Веса слоя, вручную присвоинные.
biases = np.zeros(1)            # Смещение.
model.layers[0].set_weights([weights, biases])  # Присваиваем веса и смещения
x = np.array([[1,2,3,4,5],[0,1,0,1,0]])         # Входные данные.
y = model.predict(x)
print("Enters: ", x)
print("Outputs: ", y)
print(weights)

