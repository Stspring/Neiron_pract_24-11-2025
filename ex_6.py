# import numpy as np
# from keras import models, layers


# initializators = ['random_normal', 'he_normal', 'glorot_uniform']
# for init in initializators:
#     model = models.Sequential([
#         layers.Dense(3, kernel_initializer = init, input_shape = (5,)),
#         layers.Dense(3, activation = 'softmax')
#     ])
#     w, b = model.layers[0].get_weights()
#     print(f'Initializer: {init}')
#     print(f'mean: {np.mean(w)}, std: {np.std(w)}')
#     print(f'min: {np.min(w)}, max: {np.max(w)}')


import numpy as np
from keras import models, layers

initializators = ['random_normal', 'he_normal', 'glorot_uniform']

for init in initializators:
    model = models.Sequential([
        layers.Dense(64, kernel_initializer=init, bias_initializer='zeros', input_shape=(100,)),
        layers.Dense(10, activation='softmax')
    ])
    
    w, b = model.layers[0].get_weights()

    print(f'Initializer: {init}')
    print(f'Weights - mean: {np.mean(w)}, std: {np.std(w)}')
    print(f'Weights - min: {np.min(w)}, max: {np.max(w)}')
    
    # print(f'Biases - mean: {np.mean(b)}, std: {np.std(b)}')
    # print(f'Biases - min: {np.min(b)}, max: {np.max(b)}')

