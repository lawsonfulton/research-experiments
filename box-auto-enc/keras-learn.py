import random
import math

import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.layers import Input, Dense
from keras.models import Model

def rand_box(x_bound, y_bound, size=1.0):
    """ Generates a list of 4 points corresponding to a box at a random location
    and orientation within the square defined by (-x_bound,-ybound) and (x_bound,y_bound)"""
    box = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])

    theta = random.uniform(0, math.pi * 2)
    offset = np.array([
        random.uniform(x_bound[0] + size, x_bound[1] - size),
        random.uniform(y_bound[0] + size, y_bound[1] - size)
    ])

    rotMatrix = np.array(
        [[np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]]
    )

    return np.dot(rotMatrix, box.transpose()).transpose() + offset

def draw_box(box, ax, colour='r', linewidth=None):
    poly = plt.Polygon(box, closed=True, fill=None, edgecolor=colour, linewidth=linewidth)
    ax.add_patch(poly)

scale = 1.0
def box_to_vec(box):
    return (box.flatten() / 20.0 + 0.5) * scale # normalize

def box_from_vec(box_vec):
    return ((box_vec / scale - 0.5) * 20.0).reshape((4,2))

def generate_box_samples(sample_size, x_bounds, y_bounds):
    return np.array([box_to_vec(rand_box(x_bounds, y_bounds)) for _ in range(sample_size)])

def main():
    # Setup matplotlib
    x_bounds = [-10, 10]
    y_bounds = [-10, 10]

    x_bounds = [-10, 10]
    y_bounds = [-10, 10]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_aspect('equal')

    # Setup and train the neural net
    training_sample_size = 100000
    test_sample_size = 100

    print("Generating training data...")
    # TODO This could be more efficient
    train_data = generate_box_samples(training_sample_size, x_bounds, y_bounds)
    test_data = generate_box_samples(test_sample_size, x_bounds, y_bounds)
    # TODO test data should be from a different part of the plane to make sure we are generalizing
    print("Done.")
    
    # this is the size of our encoded representations
    encoding_dim = 8

    initializer = keras.initializers.RandomUniform(minval=0.0, maxval=0.05, seed=None)

    # this is our input placeholder
    input_vec = Input(shape=(8,)) #TODO try different shapes
    #encoded = Dense(32, activation='relu')(input_vec)
    #encoded = Dense(16, activation='relu')(encoded)

    encoded = Dense(8, activation='relu', kernel_initializer=initializer)(input_vec)

    #decoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(8, activation='relu',kernel_initializer=initializer)(encoded)

    autoencoder = Model(input_vec, decoded)
    optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')

    #train
    autoencoder.fit(train_data, train_data,
                epochs=10,
                batch_size=256,#512,
                shuffle=True,
                validation_data=(test_data, test_data))

    #show
    # encode and decode some digits
    # note that we take them from the *test* set
    decoded_boxes = autoencoder.predict(test_data)

    # Draw some output
    for i in range(10):
        draw_box(box_from_vec(test_data[i]), ax, 'r', linewidth=5)
        draw_box(box_from_vec(decoded_boxes[i]), ax, 'b')

    plt.show()

if __name__ == "__main__":
    main()
