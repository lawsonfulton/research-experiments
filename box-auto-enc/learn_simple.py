import math
import time

import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Input, Dense, Dropout
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
        np_space = np.random.uniform

    length = 0.1
    
    centers = np_space(length / 2.0, 1.0 - length / 2.0, (n_samples, 2))
    thetas = np_space(0.0, 2 * np.pi, n_samples)
    
    dxs = length / 2.0 * np.sin(thetas)
    dys = length / 2.0 * np.cos(thetas)
    
    offsets = np.column_stack((dxs, dys))
    
    p1s = centers + offsets
    p2s = centers - offsets

    inputs = np.column_stack((p1s, p2s))
    outputs = np.column_stack((centers, thetas))

    return inputs, normalize_outputs(outputs)

def normalize_outputs(outputs):
    normalized_output = outputs.copy()
    normalized_output[:,2] *= 1.0 / (2.0 * np.pi)

    return normalized_output

def denormalize_outputs(outputs):
    denormalized_output = outputs.copy()
    denormalized_output[:,2] *= 2.0 * np.pi

    return denormalized_output

def draw_line(line, ax, colour='r', linewidth=None):
    poly = plt.Polygon(line.reshape(2,2), closed=False, fill=None, edgecolor=colour, linewidth=linewidth)
    ax.add_patch(poly)

def line_from_mapping(mapping):
    length = 0.1 #see above

    center = np.array(mapping[:2])
    theta = mapping[2]
    offset = np.array([length / 2.0 * np.sin(theta), length / 2.0 * np.cos(theta)])
    
    p1 = center + offset
    p2 = center - offset

    return np.append(p1, p2)

def learn_simple():
    # TODO normalize the data to [0, 1]
    train_data = get_samples(1000000)
    test_data = get_samples(100)

    # Needed for relu but apparently not for elu
    # For some reason I can't learn high frequency functions with relu alone, and the positive initializer seems
    # to mess with elu
    initializer = 'glorot_uniform'
    activation = 'elu'

    initializer = keras.initializers.RandomUniform(minval=0.0, maxval=0.01, seed=5)
    activation = 'relu'

    input = Input(shape=(len(train_data[0][0]),))
    output = Dense(50, activation=activation,)(input)
    output = Dense(50, activation=activation,)(output)
    output = Dense(50, activation=activation,)(output)
    output = Dense(50, activation=activation,)(output)
    
    # output = Dense(20, activation=activation,)(output)
    # output = Dense(20, activation=activation,)(output)
    output = Dense(len(train_data[1][0]), activation='linear', kernel_initializer=initializer, bias_initializer=initializer)(output) # First test seems to indicate no change on output with linear

    model = Model(input, output)

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error'
    )

    start = time.time()
    model.fit(
        *train_data,
        epochs=50   ,
        batch_size=4096,
        shuffle=True,
        validation_data=test_data
    )
    print("Training took: ", time.time() - start)


    viz_in, viz_actual_normalized = get_samples(200, stochastic=True)
    viz_actual = denormalize_outputs(viz_actual_normalized)
    viz_predicted = denormalize_outputs(model.predict(viz_in))
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111)


    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(223)#1,2,1)
    ax.set_aspect('equal')
    ax2 = fig.add_subplot(221)#1,2,2)
    ax3 = fig.add_subplot(222)


    for i,_ in enumerate(viz_in[:15]):
        draw_line(viz_in[i], ax, colour='r', linewidth=5)
        draw_line(line_from_mapping(viz_predicted[i]), ax, colour='b', linewidth=2)

    theta_actual = viz_actual[:,2]
    theta_predict = viz_predicted[:,2]

    sorted_indices = np.argsort(theta_actual)

    thetas = np.stack((theta_actual[sorted_indices], theta_predict[sorted_indices]))

    ax2.plot(thetas[0], np.sin(thetas[0]), color='r', linewidth=5)
    ax2.plot(thetas[0], np.sin(thetas[1]), color='b')

    ax3.plot(thetas[0], thetas[0], color='r', linewidth=5)
    ax3.plot(thetas[0], thetas[1], color='b')

    plt.show()

