import numpy as np
# import matlotlib.pyplot as plt
import tensorflow as tf 

from neural_arithmetic_logic_units.nac import NAC
from neural_arithmetic_logic_units.nalu import NALU

LR = 1e-3
ACTIVATION_FUNCTIONS = ['relu', 'tanh', 'sigmoid']
BATCH_SIZE = 1024
EPOCHS = 100
VERBOSE = 0  # no progression bar


def arithmetic_task_ds(train_size=100000, low=-5, high=5, task='+'):
    """
    task: +, -, *, /
    """
    # random subsection for a and b
    # subsections = {}
    # subsections['a'] = [0, 1]
    # subsections['b'] = [50, 51]

    x_train = (np.random.uniform(low, high, (train_size, 2))).astype(np.float32) 

    a, b = x_train[:, 0], x_train[:, 1]
    if task == '+':
        y_train = a + b
    elif task == '-':
        y_train = a - b
    elif task == '*':
        y_train = a * b 
    else:
        # division, /
        y_train = a / b
    y_train = y_train.reshape(-1, 1)
    assert x_train.shape == (train_size, 2)
    assert y_train.shape == (train_size, 1)
    return x_train, y_train

def accuracy(y_true, y_pred):
    acc = np.mean(np.isclose(y_true, y_pred, atol=1e-2, rtol=1e-2)) 
    return acc

def create_baseline_model(activation='relu'):
    inputs = tf.keras.Input(shape=(2,))
    x = tf.keras.layers.Dense(2, activation=activation)(inputs)
    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=LR), loss='mse')
    return model 


def create_stacked_nac():
    inputs = tf.keras.Input(shape=(2,))
    x = NAC(2)(inputs)
    output = NAC(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=LR), loss='mse')
    return model

def create_stacked_nalu():
    inputs = tf.keras.Input(shape=(2,))
    x = NALU(2)(inputs)
    output = NALU(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=LR), loss='mse')
    return model


if __name__ == '__main__':
    task = '+'
    x_train, y_train = arithmetic_task_ds(low=-5, high=5, task=task)
    x_test_in, y_test_in = arithmetic_task_ds(1000, low=-20, high=20, task=task)  # interpolation task    
    x_test_ex, y_test_ex = arithmetic_task_ds(1000, low=-20, high=20, task=task)  # extrapolation task

    # print(x_test_in[0][0] + x_test_in[0][1], y_test_in[0][0])
    mse_dict = {}
    x = np.array([1, 1, 0]).reshape((-1, 1))
    y = np.array([0, 0, 0]).reshape((-1, 1))

    # baseline: with different non linear activation function
    for activation in ACTIVATION_FUNCTIONS:
        model = create_baseline_model(activation=activation)
        # print(model.summary())
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=VERBOSE)
        
        # prediction
        y_pred_in = model.predict(x_test_in)
        y_pred_ex = model.predict(x_test_ex)

        # print([(y_pred_in[i][0], y_test_in[i][0]) for i in range(10)])
        mse_dict[activation] = {}

        mse_dict[activation]['mse_in'] = np.abs(y_pred_in - y_test_in)
        mse_dict[activation]['mse_ex'] = np.abs(y_pred_ex - y_test_ex)
        print('{} interpolation accuracy {} extrapolation accuracy {}'.format(activation, accuracy(y_test_in, y_pred_in), accuracy(y_test_ex, y_pred_ex)))
        
    # nac
    nac_model = create_stacked_nac()
    # nalu
    nalu_model = create_stacked_nalu()
    nac_model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=VERBOSE)
    nalu_model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=VERBOSE)

    for name, model in zip(['nac', 'nalu'], [nac_model, nalu_model]):
        y_pred_in = model.predict(x_test_in)
        y_pred_ex = model.predict(x_test_ex)

        mse_dict[name] = {}
        mse_dict[name]['mse_in'] = np.abs(y_pred_in - y_test_in)
        mse_dict[name]['mse_ex'] = np.abs(y_pred_ex - y_test_ex)
        # print(name)
        # print(y_test_in[0], y_pred_in[0])
        # print(y_test_ex[0], y_pred_ex[0])
        print('{} interpolation accuracy {} extrapolation accuracy {}'.format(name, accuracy(y_test_in, y_pred_in), accuracy(y_test_ex, y_pred_ex)))
    
    print('')
    print('*********MSE*********')
    for name ,mse in mse_dict.items():
        print('{} interpolation: {} extrapolation {}'.format(name, np.mean(mse['mse_in']), np.mean(mse['mse_ex'])))
      

    
    




    





    
    
