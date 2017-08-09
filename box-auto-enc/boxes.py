import math

import matplotlib.pyplot as plt
#import numpy as np
import autograd.numpy as np

def generate_samples(sample_size=None, x_bounds=[-10,10], y_bounds=[-10,10], size=2.0, thetas=None, offsets=None):
    """Assumes bounds are of the form x_bounds == y_bounds == [-b, b]"""
    if sample_size is None:
        sample_size = len(thetas) if thetas is not None else len(offsets)

    half_size = size / 2.0
    box = np.array([[-half_size, half_size], [half_size, half_size], [half_size, -half_size], [-half_size, -half_size]]).transpose()
    bound_buffer = math.sqrt(2 * (size / 2.0) ** 2) # Squares rotate...

    if thetas is not None:
        theta = np.array(thetas)
    else:
        theta = np.random.uniform(0, math.pi * 2, sample_size)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    if offsets is not None:
        offset = np.array([[v[0] for v in offsets], [v[1] for v in offsets]]).transpose().reshape(-1,2,1) # WARNING slow
    else:
        offset = np.array([
            np.random.uniform(x_bounds[0] + bound_buffer, x_bounds[1] - bound_buffer, sample_size),
            np.random.uniform(y_bounds[0] + bound_buffer, y_bounds[1] - bound_buffer, sample_size)
        ]).transpose().reshape(-1,2,1)

    rotMatrices = np.array(
        [[cos_theta, -sin_theta],
        [sin_theta,  cos_theta]]
    ).transpose() # is this fast?

    rotated_boxes = np.dot(rotMatrices, box) + offset
    normalized_boxes = rotated_boxes / (x_bounds[1] - x_bounds[0]) + 0.5 # See docstring for assumption

    return normalized_boxes.transpose((0,2,1)).reshape(sample_size, 8)

def explicit_decode(q, size=2.0):
    """q = (x, y, theta)"""
    center = q[:2]
    theta = q[2]

    return generate_samples(thetas=np.array([theta]), offsets=np.array([center]))[0]


def draw_box(box, ax, colour='r', linewidth=None):
    poly = plt.Polygon(box, closed=True, fill=None, edgecolor=colour, linewidth=linewidth, facecolor='y')
    return ax.add_patch(poly)

def draw_from_samples(ax, box_samples, colour='r', linewidth=None):
    for b in box_samples:
        draw_box(box_from_vec(b), ax, colour, linewidth=linewidth)

scale = 1.0
def box_to_vec(box):
    return (box.flatten() / 20.0 + 0.5) * scale # normalize

def box_from_vec(box_vec):
    return ((box_vec / scale - 0.5) * 20.0).reshape((4,2))
