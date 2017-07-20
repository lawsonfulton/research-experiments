import random
import math

import matplotlib.pyplot as plt
import numpy as np

import nn

# Setup and train the neural net
training_sample_size = 1000000
test_sample_size = 1000

print("Generating training data...")
# TODO This could be more efficient
train_data = np.array()
test_data = generate_box_samples(test_sample_size, x_bounds, y_bounds)
print("Done.")

# Layer sizes
vec_size = len(train_data[0])
assert vec_size == 8
layer_widths = [vec_size, 100, 3, 100, vec_size]

# Optimization hyperparameters
param_scale = 0.1
batch_size = 256 # TODO does batch size effect learning rate?
num_epochs = 8
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