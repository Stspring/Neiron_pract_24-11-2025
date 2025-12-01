import numpy as np
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical

(x_train, y_train), _ = mnist.load_data()
x_train = x_train[:200].reshape((200, 28*28)).astype('float32') / 255
y_train = to_categorical(y_train[:200])

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(28*28,)),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy')
w_before, b_before = model.layers[0].get_weights()
model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1)
w_after, b_after = model.layers[0].get_weights()
delta_w = w_after - w_before

flat_delta_w = delta_w.flatten()
top10_indices = np.argsort(np.abs(flat_delta_w))[-10:][::-1]
top10_weights = flat_delta_w[top10_indices]

print('Топ-10 изменений весов:')
for idx, val in zip(top10_indices, top10_weights):
    print(f'Индекс: {idx}, Значение изменения: {val}')