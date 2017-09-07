import math
import time
import datetime

# import matplotlib as mpl
# mpl.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import keras
import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model, Sequential, load_model
from keras import regularizers
import tensorflow as tf

import pendulums

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
    training_sample_size = 500000#2**20
    test_sample_size = 2**14

    print("Generating training data...")


    #train_data = pendulums.generate_samples(training_sample_size)

    theta1_granularity = 1000
    theta1 = np.linspace(0.0, 2*math.pi, num=theta1_granularity)
    theta2_granularity = 1000
    theta2 = np.linspace(0.0, 2*math.pi, num=theta2_granularity)
    thetas = np.transpose([np.tile(theta1, len(theta2)), np.repeat(theta2, len(theta1))])

    train_data = pendulums.generate_samples(thetas=thetas)
    test_data = pendulums.generate_samples(test_sample_size, x_bounds, y_bounds)

    # TODO test data should be from a different part of the plane to make sure we are generalizing
    print("Done. Runtime: ", time.time()-start_time)
    
    model_start_time = time.time()

    initializer = keras.initializers.RandomUniform(minval=0.0001, maxval=0.01)
    # bias_initializer = initializer
    activation = 'relu' #keras.layers.advanced_activations.LeakyReLU(alpha=0.3) #'relu'


    # autoencoder = Model(input, output)
    num_layers = 3
    layer_size = 1024

    #{'activation': 'relu', 'activation_1': 'sigmoid', 'batch_size': 512, 'layer_size': 1024, 'lr': 0.0005, 'num_layers': 3}

    # # Single autoencoder
    # input = Input(shape=(len(train_data[0]),))
    # output = input

    # output = Dense(32, activation=activation)(output)
    # output = Dense(512, activation=activation)(output)
    # output = Dense(32, activation=activation)(output)

    # output = Dense(2, activation='sigmoid', name="encoded", kernel_initializer=initializer, bias_initializer='zeros')(output) #maybe just try avoiding dead neurons in small encoding layer?
    
    # output = Dense(32, activation=activation)(output)
    # output = Dense(512, activation=activation)(output)
    # output = Dense(32, activation=activation)(output)

    # output = Dense(len(train_data[0]), activation='sigmoid')(output)#'linear',)(output) # First test seems to indicate no change on output with linear

    # autoencoder = Model(input, output)

    input = Input(shape=(len(train_data[0]),))
    output = input

    output = Dense(32, activation=activation)(output)
    output = Dense(512, activation=activation)(output)
    output = Dense(1024, activation=activation)(output)

    output = Dense(100, activation='linear', name="encoded", activity_regularizer=regularizers.l1(10e-5), kernel_initializer=initializer, bias_initializer='zeros')(output) #maybe just try avoiding dead neurons in small encoding layer?
    
    output = Dense(1024, activation=activation)(output)
    output = Dense(512, activation=activation)(output)
    output = Dense(32, activation=activation)(output)

    output = Dense(len(train_data[0]), activation='linear')(output)#'linear',)(output) # First test seems to indicate no change on output with linear

    autoencoder = Model(input, output)


    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    autoencoder.compile(
        optimizer= optimizer, #adamax and nadam worked well too
        loss="mean_squared_error",#custom_loss(autoencoder)#
    )

    start = time.time()

    # from keras.callbacks import TensorBoard
    # tensorboard = TensorBoard(log_dir="logs/{}".format(datetime.datetime.now().strftime("%I%M%p%B%d%Y")), write_graph=False)#,histogram_freq=2,)#, histogram_freq=1, write_graph=False)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='min')
    autoencoder.fit(
        train_data, train_data,
        epochs=20,
        batch_size=512,
        shuffle=True,
        callbacks=[early_stop],#tensorboard],
        validation_data=(test_data, test_data)
    )

    output_path = 'models/' + datetime.datetime.now().strftime("%I %M%p %B %d %Y") + '.h5'
    autoencoder.save(output_path)
    print("Saved at:", output_path)

    ### MY stacked autoencoder
    print("Total model time: ", time.time() - model_start_time)

    #show
    # encode and decode some digits
    # note that we take them from the *test* set
    predict_start = time.time()
    test_data = test_data
    decoded_pendulums = autoencoder.predict(test_data)
    print('Predict took: ', time.time() - predict_start)
    
    print("Total runtime: ", time.time() - start_time)

    # Draw some output
    num_to_show = 15
    pendulums.draw_from_samples(ax, test_data[:num_to_show], 'r', linewidth=2)
    pendulums.draw_from_samples(ax, decoded_pendulums[:num_to_show], 'b',linewidth=1)

    plt.show()

if __name__ == "__main__":
    main()
