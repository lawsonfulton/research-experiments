import math
import time

import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Input, Dense
from keras.models import Model, Sequential, load_model
from keras import regularizers

def get_samples_sin(n_samples, stochastic=True):
    domain = [0.0, 1.0]#2*np.pi]

    if stochastic:
        inputs = np.random.uniform(*domain, n_samples)
    else:
        inputs = np.linspace(*domain, n_samples)

    outputs = (np.sin(inputs * 2 * np.pi * 2) + 1.0) / 2.0

    return inputs, outputs

def get_samples(n_samples, stochastic=True):
    if stochastic:
        np_space = np.random.uniform
    else:
        np_space = np.linspace

    length = 0.1
    
    grid_size = [n_samples, 2]
    centers = [np_space(0.0, 1.0 - length / 2.0, g) for g in grid_size]

    # x_centers = np_space(0.0, 1.0 - length / 2.0, n_samples)
    # y_centers = np_space(0.0, 1.0 - length / 2.0, n_samples)
    centers = np.mesh_grid(x_centers, y_centers)

    thetas = np_space(0.0, 2 * math.pi, n_samples)
    
    dxs = length / 2.0 * np.sin(thetas)
    dys = length / 2.0 * np.cos(thetas)
    
    offsets = np.column_stack((dxs, dys))
    
    p1s = centers + offsets
    p2s = centers - offsets

    inputs = np.column_stack((p1s, p2s))
    outputs = np.column_stack((centers, thetas))

    return inputs, outputs

def learn_simple():
    # TODO normalize the data to [0, 1]
    train_data = get_samples(100000)
    test_data = get_samples(100)

    # Needed for relu but apparently not for elu
    # For some reason I can't learn high frequency functions with relu alone, and the positive initializer seems
    # to mess with elu
    initializer = 'glorot_uniform'#keras.initializers.RandomUniform(minval=0.0, maxval=0.1, seed=None)
    activation = 'elu'

    input = Input(shape=(len(train_data[0][0]),))
    output = Dense(10, activation=activation, kernel_initializer=initializer, bias_initializer=initializer)(input)
    output = Dense(10, activation=activation, kernel_initializer=initializer, bias_initializer=initializer)(output)
    output = Dense(10, activation=activation, kernel_initializer=initializer, bias_initializer=initializer)(output)
    output = Dense(len(train_data[1][0]), activation=activation, kernel_initializer=initializer, bias_initializer=initializer)(output) # First test seems to indicate no change on output with linear

    model = Model(input, output)

    optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error'
    )

    start = time.time()
    model.fit(
        *train_data,
        epochs=3,
        batch_size=512,
        shuffle=True,
        validation_data=test_data
    )
    print("Training took: ", time.time() - start)


    viz_in, viz_actual = get_samples(10, stochastic=False)
    viz_predicted = model.predict(viz_in)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    plt.plot(viz_in, viz_actual)
    plt.plot(viz_in, viz_predicted, color='r')

    plt.show()

