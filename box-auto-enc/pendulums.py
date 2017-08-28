import math

import matplotlib.pyplot as plt
from matplotlib import collections
import numpy as np
# Need to use autograd numpy if we are using autograd for gradients
#import autograd.numpy as np


def generate_samples(sample_size=None, x_bounds=[-10,10], y_bounds=[-10,10], arm_lengths=[4.0, 4.0], thetas=None):
    """Assumes bounds are of the form x_bounds == y_bounds == [-b, b]
    thetas=[[theta1, theta2], ...]
    Pendulum will be anchored at (0,0) so can be described as [(x1,y1), (x2,y2)]
    """
    origin = np.array([0.0, 0.0])
    refrence_arms = np.array([[arm_lengths[0], 0.0], [arm_lengths[1], 0.0]])

    if sample_size is None:
        sample_size = len(thetas) if thetas is not None else len(offsets)

    if thetas is not None:
        thetas = np.array(thetas)
    else:
        thetas = np.random.uniform(0, np.pi * 2, (sample_size, 2))

    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)

    rotMatrices = np.array(
        [[cos_theta, -sin_theta],
        [sin_theta,  cos_theta]]
    ).transpose() # (2, n_samples) list of rotation matrices rotMatrices[0][i] -> First arm, ith sample, rotMatrices[1] -> second arm, ith sample

    rotated_arms = np.array([
        np.zeros((sample_size, 2)),
        np.dot(rotMatrices[0], refrence_arms[0]),
        np.dot(rotMatrices[1], refrence_arms[1])
    ])
    rotated_arms[2] = rotated_arms[1] + rotated_arms[2] 
    
    #normalized_arms = rotated_arms
    normalized_arms = rotated_arms / (x_bounds[1] - x_bounds[0]) + 0.5 # See docstring for assumption

    return normalized_arms.transpose((1,0,2)).reshape(sample_size, 6)

# def explicit_decode(q, size=2.0):
#     """q = (x, y, theta)"""
#     center = q[:2]
#     theta = q[2] 

#     return generate_samples(thetas=np.array([theta]), offsets=np.array([center]))[0]


def draw_pendulum(pendulum, ax, colour='r', linewidth=1):
    poly = plt.Polygon(pendulum, closed=False, fill=None, edgecolor=colour, linewidth=linewidth, facecolor='y')
    points, = ax.plot(pendulum[:,0], pendulum[:,1], colour + 'o', ms=linewidth * 3)
    return ax.add_patch(poly)

def draw_from_samples(ax, pendulum_samples, colour='r', linewidth=2):
    # c = collections.LineCollection([pendulum_from_vec(b) for b in pendulum_samples], colors='r', linewidths=2)
    # ax.add_collection(c)
    for b in pendulum_samples:
        draw_pendulum(pendulum_from_vec(b), ax, colour, linewidth=linewidth)

scale = 1.0
def pendulum_to_vec(pendulum):
    return (pendulum.flatten() / 20.0 + 0.5) * scale # normalize

def pendulum_from_vec(pendulum_vec):
    return ((pendulum_vec / scale - 0.5) * 20.0).reshape((3,2))


def tests():
    eps = 0.0000001

    arm_length = 1.0
    samples = generate_samples(100, arm_lengths=[arm_length, arm_length])

    for sample in samples:
        assert np.linalg.norm(sample - pendulum_to_vec(pendulum_from_vec(sample))) < eps

        # Check arm lengths
        assert np.linalg.norm(pendulum_from_vec(sample)[1]) - arm_length < eps
        assert np.linalg.norm(pendulum_from_vec(sample)[2] - pendulum_from_vec(sample)[1]) - arm_length < eps

    print("Passed.")
