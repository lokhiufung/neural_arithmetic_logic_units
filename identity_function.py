import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


LR = 0.01
ACTIVATION_FUNCTIONS = ['relu', 'tanh', 'sigmoid']
BATCH_SIZE = 128
EPOCHS = 1

def identity_ds(train_size=100000, low=0.0, high=1.0):
  # identity function with given range
  x_train = np.float32(np.random.uniform(low, high, (train_size, 1)))
  y_train = x_train.copy()
  return x_train, y_train

def create_model(activation='relu'):
    inputs = tf.keras.Input(shape=(1,))
    x = tf.keras.layers.Dense(8, activation=activation)(inputs)
    x = tf.keras.layers.Dense(8, activation=activation)(x)
    x = tf.keras.layers.Dense(8, activation=activation)(x)
    x = tf.keras.layers.Dense(8, activation=activation)(x)
    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=LR), loss='mse')
    return model


if __name__ == '__main__':
    ######
    # Show the numerical extrapolation failure in neural network
    # non linearity of activation function drastically affect the linear extrapolation
    # relu is better here due to its "partially linear" shape...  
    train_range = [-5, 5]
    test_range = [-20, 20]

    x_train, y_train = identity_ds(low=train_range[0], high=train_range[1])
    x_test, y_test = identity_ds(10000, low=test_range[0], high=test_range[1])
    
    mae_dict = {}
    for activation in ACTIVATION_FUNCTIONS:
        model = create_model(activation=activation)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
        
        # prediction
        y_pred = model.predict(x_test)
        mae_dict[activation] = {}
        mae_dict[activation]['mae'] = np.abs(y_pred - y_test)
        mae_dict[activation]['x'] = x_test 

    # show mae vs x
    fig, ax = plt.subplots()
    for activation in ACTIVATION_FUNCTIONS:
        ax.scatter(mae_dict[activation]['x'], mae_dict[activation]['mae'], label=activation)
    ax.legend()
    ax.axvline(x=train_range[0], color='red')
    ax.axvline(x=train_range[1], color='red')
    ax.set_title('Training range: [{}, {}]'.format(train_range[0], train_range[1]))
    ax.set_xlabel('x')
    ax.set_ylabel('mean absolute error')
    ax.grid()
    plt.show()
    




    