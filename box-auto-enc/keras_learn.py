import math
import time
import datetime

# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import keras
import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model, Sequential, load_model
from keras import regularizers

import boxes
import keras_autoencoder as autoencoder

def add_noise(samples):
    return samples #+ np.random.normal(0.0, 0.005, (len(samples), len(samples[0])))

def explore_model(path):
    print("Loading model...")
    model = load_model(path)
    print("Done.")

    ##TODO  ~~~~~~
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded').output)

    return model

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
    training_sample_size = 1000000
    test_sample_size = 100

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

    # Needed for relu but apparently not for elu
    # For some reason I can't learn high frequency functions with relu alone, and the positive initializer seems
    # to mess with elu
    # initializer = 'glorot_uniform'
    # activation = 'elu'

    initializer = keras.initializers.RandomUniform(minval=0.0, maxval=0.01, seed=5)
    bias_initializer = initializer
    activation = 'relu'

    # , kernel_initializer=initializer, bias_initializer=initializer
    # , kernel_initializer=initializer, bias_initializer=initializer
    # , kernel_initializer=initializer, bias_initializer=initializer
    # , kernel_initializer=initializer, bias_initializer=initializer
    input = Input(shape=(len(train_data[0]),))
    output = Dense(200, activation=activation)(input)
    output = Dense(100, activation=activation)(output)
    output = Dense(3, activation=activation, name="encoded")(output)
    output = Dense(100, activation=activation)(output)
    output = Dense(200, activation=activation)(output)
    output = Dense(len(train_data[0]), activation='linear')(output)#'linear',)(output) # First test seems to indicate no change on output with linear

    autoencoder = Model(input, output)
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded').output)

    def contractive_loss(y_pred, y_true):
        lam = 1e-4
        mse = K.mean(K.square(y_true - y_pred), axis=1)

        W = K.variable(value=autoencoder.get_layer('encoded').get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = autoencoder.get_layer('encoded').output
        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

        return mse + contractive

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    autoencoder.compile(
        optimizer=optimizer,
        loss='mean_squared_error'
    )

    start = time.time()

    autoencoder.fit(
        add_noise(train_data), train_data,
        epochs=50  ,
        batch_size=8192,
        shuffle=True,
        validation_data=(test_data, test_data)
    )
    output_path = 'models/' + datetime.datetime.now().strftime("%I %M%p %B %d %Y") + '.h5'
    autoencoder.save(output_path)

    # output_path = 'models/' + datetime.datetime.now().strftime("%I %M%p %B %d %Y") + '.h5'
    # layer_dims = [box_dim, 20,  7]
    # model = autoencoder.train_model(
    #     train_data,
    #     test_data,
    #     layer_dims=layer_dims,
    #     learning_rate=0.001,
    #     epochs=5,
    #     batch_size=4096,
    #     loss='mean_squared_error',
    #     saved_model_path=output_path
    # )
    print("Total model time: ", time.time() - model_start_time)

    #show
    # encode and decode some digits
    # note that we take them from the *test* set
    predict_start = time.time()
    test_data = add_noise(test_data)
    decoded_boxes = autoencoder.predict(test_data)
    print('Predict took: ', time.time() - predict_start)
    
    print("Total runtime: ", time.time() - start_time)

    # Draw some output
    num_to_show = 15
    boxes.draw_from_samples(ax, test_data[:num_to_show], 'r', linewidth=5)
    boxes.draw_from_samples(ax, decoded_boxes[:num_to_show], 'b',linewidth=2)

    plt.show()

if __name__ == "__main__":
    main()
