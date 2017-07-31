import math
import time
import datetime

# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import boxes
import keras_autoencoder as autoencoder


def main():
    start_time = time.time()
    # Setup matplotlib
    x_bounds = [-10, 10]
    y_bounds = [-10, 10]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_aspect('equal')

    # Setup and train the neural net
    training_sample_size = 500000
    test_sample_size = 1000

    print("Generating training data...")
    # TODO This could be more efficient
    #train_data = generate_box_samples(training_sample_size, x_bounds, y_bounds)
    train_data = boxes.generate_samples(training_sample_size)
    test_data = boxes.generate_samples(test_sample_size, x_bounds, y_bounds)
    # TODO test data should be from a different part of the plane to make sure we are generalizing
    print("Done. Runtime: ", time.time()-start_time)
    
    model_start_time = time.time()
    # this is the size of our encoded representations
    encoding_dim = 8
    box_dim = 8

    output_path = 'models/' + datetime.datetime.now().strftime("%I %M%p %B %d %Y") + '.h5'
    layer_dims = [box_dim, 200, encoding_dim]
    model = autoencoder.train_model(
        train_data,
        test_data,
        layer_dims=layer_dims,
        learning_rate=0.01,
        epochs=5,
        batch_size=2048,
        loss='mean_squared_error',
        saved_model_path=output_path
    )
    print("Total model time: ", time.time() - model_start_time)

    #show
    # encode and decode some digits
    # note that we take them from the *test* set
    predict_start = time.time()
    decoded_boxes = model.predict(test_data)
    print('Predict took: ', time.time() - predict_start)
    
    print("Total runtime: ", time.time() - start_time)

    # Draw some output
    num_to_show = 15
    boxes.draw_from_samples(ax, test_data[:num_to_show], 'r', linewidth=5)
    boxes.draw_from_samples(ax, decoded_boxes[:num_to_show], 'b')

    plt.show()

if __name__ == "__main__":
    main()
