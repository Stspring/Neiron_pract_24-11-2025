import numpy as np
from keras import models, layers

model = models.Sequential([
    layers.Dense (3, input_shape = (5,), activation="relu")     # Создаем слой
])

weight = input('Введите значение веса ')
weights = np.full((5,3), weight)     # Веса слоя, вручную присвоинные.
biases = np.zeros()            # Смещение.
model.layers[0].set_weights([weights, biases])  # Присваиваем веса и смещения
x = np.array([[1,2,3,4,5],[0,1,0,1,0]])         # Входные данные.
y = model.predict(x)
print("Enters: ", x)
print("Outputs: ", y)
print(weights)

# model = models.Sequential([
#     (layers.Dense(2, input_shape=(3,)))
# ])

# weights, biases = models.layers[0].get_weights()
# weights.fill(1)
# model.set_weights([weights, biases])

# input_data = np.array([[1, 2, 3]])
# output = model.predict(input_data)

# print("Входные данные:", input_data)
# print("Матрица весов:\n", weights)
# print("Вывод слоя:", output)