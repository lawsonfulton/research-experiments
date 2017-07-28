import math
import time

import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Input, Dense
from keras.models import Model, Sequential, load_model
from keras import regularizers

def get_samples(n_samples, stochastic=True):
    domain = [0.0, 1.0]#2*np.pi]

    if stochastic:
        inputs = np.random.uniform(*domain, n_samples)
    else:
        inputs = np.linspace(*domain, n_samples)

    outputs = np.sin(inputs * 2 * np.pi * 2) + 1.0

    return inputs, outputs

def learn_simple():
    # TODO normalize the data to [0, 1]
    train_data = get_samples(500000)
    test_data = get_samples(100)

    # Needed for relu but apparently not for elu
    # For some reason I can't learn high frequency functions with relu alone, and the positive initializer seems
    # to mess with elu
    # initializer = keras.initializers.RandomUniform(minval=0.0, maxval=0.05, seed=None)
    activation = 'elu'

    input = Input(shape=(1,))
    output = Dense(2, activation=activation)(input)
    output = Dense(10, activation=activation)(output)
    output = Dense(2, activation=activation)(output)
    output = Dense(1, activation=activation)(output) # First test seems to indicate no change on output with linear

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


    viz_in, viz_actual = get_samples(50, stochastic=False)
    viz_predicted = model.predict(viz_in)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    plt.plot(viz_in, viz_actual)
    plt.plot(viz_in, viz_predicted, color='r')

    plt.show()

