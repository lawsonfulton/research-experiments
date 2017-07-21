import random
import math
import time

import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers

# TODO make sure nothing is generate out of bounds
def rand_box(x_bounds, y_bounds, size=2.0):
    """ Generates a list of 4 points corresponding to a box at a random location
    and orientation within the square defined by x_bounds and y_bounds with side length size"""
    half_size = size / 2.0
    box = np.array([[-half_size, half_size], [half_size, half_size], [half_size, -half_size], [-half_size, -half_size]])

    bound_buffer = math.sqrt(2 * (size / 2.0) ** 2) # Squares rotate...
    theta = random.uniform(0, math.pi * 2)
    offset = np.array([
        random.uniform(x_bounds[0] + bound_buffer, x_bounds[1] - bound_buffer),
        random.uniform(y_bounds[0] + bound_buffer, y_bounds[1] - bound_buffer)
    ])

    rotMatrix = np.array(
        [[np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]]
    )

    return np.dot(rotMatrix, box.transpose()).transpose() + offset


def generate_box_samples(sample_size, x_bounds, y_bounds):
    return np.array([box_to_vec(rand_box(x_bounds, y_bounds)) for _ in range(sample_size)])


def draw_box(box, ax, colour='r', linewidth=None):
    poly = plt.Polygon(box, closed=True, fill=None, edgecolor=colour, linewidth=linewidth)
    ax.add_patch(poly)

scale = 1.0
def box_to_vec(box):
    return (box.flatten() / 20.0 + 0.5) * scale # normalize

def box_from_vec(box_vec):
    return ((box_vec / scale - 0.5) * 20.0).reshape((4,2))

def main():
    start_time = time.time()
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
    training_sample_size = 1000000
    test_sample_size = 1000

    print("Generating training data...")
    # TODO This could be more efficient
    train_data = generate_box_samples(training_sample_size, x_bounds, y_bounds)
    test_data = generate_box_samples(test_sample_size, x_bounds, y_bounds)
    # TODO test data should be from a different part of the plane to make sure we are generalizing
    print("Done. Runtime: ", time.time()-start_time)
    
    model_start_time = time.time()
    # this is the size of our encoded representations
    encoding_dim = 3
    box_dim = 8

    initializer = keras.initializers.RandomUniform(minval=0.0, maxval=0.05, seed=None)

    model = Sequential()
    # Input

    model.add(Dense(200, input_shape=(box_dim,), activation='relu', kernel_initializer=initializer))

    # Encoded layer
    model.add(Dense(encoding_dim, activation='relu', kernel_initializer=initializer))
    ##

    model.add(Dense(200, activation='relu', kernel_initializer=initializer))

    # Output layer
    model.add(Dense(box_dim, activation='relu', kernel_initializer=initializer))


    optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    #train
    model.fit(train_data, train_data,
                epochs=10,
                batch_size=512, # 512
                shuffle=True,
                validation_data=(test_data, test_data))

    #show
    # encode and decode some digits
    # note that we take them from the *test* set
    decoded_boxes = model.predict(test_data)

    print("Total model time: ", time.time() - model_start_time)
    print("Total runtime: ", time.time() - start_time)

    # Draw some output
    for i in range(10):
        draw_box(box_from_vec(test_data[i]), ax, 'r', linewidth=5)
        draw_box(box_from_vec(decoded_boxes[i]), ax, 'b')

    for i in range(len(test_data)):
        if any(c > 1.0 or c < -1.0 for c in test_data[i]):
            print("Bad test data: ", box_from_vec(test_data[i]))
        if any(c > 1.0 or c < -1.0 for c in decoded_boxes[i]):
            print("Bad predicted data: ", box_from_vec(decoded_boxes[i]))

    plt.show()

if __name__ == "__main__":
    main()
