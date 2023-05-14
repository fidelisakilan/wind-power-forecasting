import random
import numpy as np
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