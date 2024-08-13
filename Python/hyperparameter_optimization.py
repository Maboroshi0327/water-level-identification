from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, quniform

import numpy as np
from tensorflow.keras import models, layers

from My_Functions import train_data, test_data


def data():
    x_train, y_train, x_test, y_test = 1, 2, 3, 4
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    sampleRate = {{choice([10000, 20000, 30000, 40000, 50000])}}
    layer_num = {{choice([2, 3])}}
    node1 = {{choice([10, 20, 30, 40, 50, 60])}}
    node2 = {{choice([10, 20, 30, 40, 50, 60])}}
    node3 = {{choice([10, 20, 30, 40, 50, 60])}}
    activation = {{choice(['relu', 'tanh'])}}
    dropout = {{choice([0.1, 0.2, 0.3])}}

    x_train, y_train = train_data(sampleRate)
    x_test, y_test = test_data(sampleRate)
    layer_num = int(layer_num)
    node1 = int(node1)
    node2 = int(node2)
    node3 = int(node3)
    # -------------------------model--------------------------
    model = models.Sequential()

    model.add(layers.Dense(node1, input_shape=(1249, )))
    model.add(layers.Activation(activation))

    model.add(layers.Dense(node2))
    model.add(layers.Activation(activation))

    if layer_num == 3:
        model.add(layers.Dense(node3))
        model.add(layers.Activation(activation))

    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(5))
    model.add(layers.Activation('softmax'))
    # --------------------------------------------------------

    # ------------------------training------------------------
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    result = model.fit(x_train, y_train, batch_size=1, epochs=5, verbose=0)
    model.save(f'./models/{sampleRate}_{layer_num}_{node1}_{node2}_{node3}_{activation}_{dropout}.h5')
    # --------------------------------------------------------

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print('--------------------------------')
    print(f'sampleRate:{sampleRate}')
    print(f'layer_num:{layer_num}')
    print(f'node1:{node1}')
    print(f'node2:{node2}')
    print(f'node3:{node3}')
    print(f'activation:{activation}')
    print(f'dropout:{dropout}')
    print(f'acc:{acc}')
    print(f'loss:{loss}')
    print('--------------------------------')

    path = 'Hyperparameter.csv'
    with open(path, 'a') as f:
        f.write(f'{sampleRate},{layer_num},{node1},{node2},{node3},{activation},{dropout},{loss},{acc}\n')

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    path = 'Hyperparameter.csv'
    with open(path, 'w') as f:
        f.write('sampleRate,layer_num,node1,node2,node3,activation,dropout,loss,acc\n')
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
   
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
