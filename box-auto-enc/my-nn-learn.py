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

def draw_boxes_from_samples(ax, box_samples, colour='r', linewidth=None):
    for b in box_samples:
        draw_box(box_from_vec(b), ax, colour, linewidth=linewidth)

scale = 1.0
def box_to_vec(box):
    return (box.flatten() / 20.0 + 0.5) * scale # normalize

def box_from_vec(box_vec):
    return ((box_vec / scale - 0.5) * 20.0).reshape((4,2))

def generate_box_samples_fast(sample_size, x_bounds=[-10,10], y_bounds=[-10,10], size=2.0):
    """Assumes bounds are of the form x_bounds == y_bounds == [-b, b]"""

    half_size = size / 2.0
    box = np.array([[-half_size, half_size], [half_size, half_size], [half_size, -half_size], [-half_size, -half_size]]).transpose()
    bound_buffer = math.sqrt(2 * (size / 2.0) ** 2) # Squares rotate...

    theta = np.random.uniform(0, math.pi * 2, sample_size)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    offset = np.array([
        np.random.uniform(x_bounds[0] + bound_buffer, x_bounds[1] - bound_buffer, sample_size),
        np.random.uniform(y_bounds[0] + bound_buffer, y_bounds[1] - bound_buffer, sample_size)
    ]).transpose().reshape(-1,2,1)

    rotMatrices = np.array(
        [[cos_theta, -sin_theta],
        [sin_theta,  cos_theta]]
    )   .transpose() # is this fast?

    rotated_boxes = np.dot(rotMatrices, box) + offset
    normalized_boxes = rotated_boxes / (x_bounds[1] - x_bounds[0]) + 0.5 # See docstring for assumption

    return normalized_boxes.transpose((0,2,1)).reshape(sample_size, 8)

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
    training_sample_size = 500000
    test_sample_size = 10

    print("Generating training data...")
    # TODO This could be more efficient
    train_data = generate_box_samples_fast(training_sample_size, x_bounds, y_bounds)
    test_data = generate_box_samples_fast(test_sample_size, x_bounds, y_bounds)

    # TODO test data should be from a different part of the plane to make sure we are generalizing
    
    print("Done.")
    
    # Layer sizes
    vec_size = len(train_data[0])
    assert vec_size == 8
    layer_widths = [vec_size, 200, 3, 200, vec_size]

    # Optimization hyperparameters
    param_scale = 0.1 #??
    batch_size = 512 # TODO does batch size effect learning rate?
    num_epochs = 20
    step_size = 0.01

    params = nn.init_params(layer_widths, param_scale)

    print("Training")
    params = nn.train(train_data, test_data, layer_widths, step_size, num_epochs, batch_size)
    with open("./models/my-nn-model.bin", 'wb') as f:
        np.save(f, params)
    print("Done.")

    # Draw some output
    # Draw some output
    decoded_boxes = nn.evaluate_net(test_data, params)
    num_to_show = 15

    draw_boxes_from_samples(ax, test_data, 'r', linewidth=5)
    draw_boxes_from_samples(ax, decoded_boxes, 'b')

    plt.show()

if __name__ == "__main__":
    main()
