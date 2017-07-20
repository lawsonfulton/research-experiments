import random
import math

import matplotlib.pyplot as plt
import numpy as np

import nn

def rand_box(x_bound, y_bound, size=1.0):
    """ Generates a list of 4 points corresponding to a box at a random location
    and orientation within the square defined by (-x_bound,-ybound) and (x_bound,y_bound)"""
    box = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])

    theta = random.uniform(0, math.pi * 2)
    offset = np.array([
        random.uniform(x_bound[0], x_bound[1]),
        random.uniform(y_bound[0], y_bound[1])
    ])

    rotMatrix = np.array(
        [[np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]]
    )

    return np.dot(rotMatrix, box.transpose()).transpose() + offset

def draw_box(box, ax, colour='r', linewidth=None):
    poly = plt.Polygon(box, closed=True, fill=None, edgecolor=colour, linewidth=linewidth)
    ax.add_patch(poly)

def box_to_vec(box):
    return box.flatten() + 11 # TODO Figure out why I need to keep everything positive..

def box_from_vec(box_vec):
    return (box_vec - 11).reshape((4,2))

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
    test_sample_size = 1000

    print("Generating training data...")
    # TODO This could be more efficient
    train_data = generate_box_samples(training_sample_size, x_bounds, y_bounds)
    test_data = generate_box_samples(test_sample_size, x_bounds, y_bounds)

    # TODO test data should be from a different part of the plane to make sure we are generalizing
    
    print("Done.")
    
    # Layer sizes
    vec_size = len(train_data[0])
    assert vec_size == 8
    layer_widths = [vec_size, 8, vec_size]

    # Optimization hyperparameters
    param_scale = 0.1
    batch_size = 256 # TODO does batch size effect learning rate?
    num_epochs = 5
    step_size = 0.01

    params = nn.init_params(layer_widths, param_scale)

    print("Training")
    params = nn.train(train_data, test_data, layer_widths, step_size, num_epochs, batch_size)
#     with open("./mnist_auto_enc_params.bin", 'wb') as f:
#         np.save(f, params)
    print("Done.")

    # Draw some output
    for i in range(10):
        input = test_data[i]
        output = nn.evaluate_net(input, params)
        
        draw_box(box_from_vec(input), ax, 'r', linewidth=5)
        draw_box(box_from_vec(output), ax, 'b')

    plt.show()

if __name__ == "__main__":
    main()
