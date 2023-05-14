import random
import numpy as np
import tensorflow as tf
ITERATIONS = 60

param_grid = {

    'NFOLDS'    : [5],

    'EPOCHS'    : [10,20],

    'neurons'   : [(100, 100), (50, 100), (100, 200)],

    'batch_size': [32, 64, 128, 256],

    'activation' : ['relu', 'leakyrelu'],

    'optimizer' : ['SGD', 'ADAM'],

    'dropout' :  np.linspace(0.1, 0.5, 5),

    'filters' : [40, 50, 60],

    'kernel_size': [2,3],

    'strides': [2,3]

    }



for key, value in param_grid.items():

    exec(key + '= value[random.randint(0, len(value) - 1)]')

    print(f'{key} = {eval(key)}')
    
    tf.keras.backend.clear_session()
    multivariate_cnn = tf.keras.models.Sequential([

        Conv1D(filters=48, kernel_size=2,

            strides=1, padding='causal',

            activation='relu', input_shape=input_shape),

        Flatten(),

        Dense(48, activation='relu'),

        Dense(1)])



    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('multivariate_cnn.h5', save_best_only=True)

    optimizer = tf.keras.optimizers.Adam(lr=6e-3, amsgrad=True)

    multivariate_cnn.compile(loss=loss,

                            optimizer=optimizer,

                            metrics=metric)