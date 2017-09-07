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
import functools
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional, loguniform, lognormal


def contractive_loss(y_pred, y_true, mdl):
    lam = 1e-3
    mse = K.mean(K.square(y_true - y_pred), axis=1)

    W = K.variable(value=mdl.get_layer('encoded').get_weights()[0])  # N x N_hidden
    W = K.transpose(W)  # N_hidden x N
    h = mdl.get_layer('encoded').output
    dh = h * (1 - h)  # N_batch x N_hidden

    # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
    contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

    return mse + contractive

def model(train_data, test_data):

    initializer = keras.initializers.RandomUniform(minval=0.0001, maxval=0.01)
    activation = 'relu'


    num_layers = {{quniform(1, 5, 1)}}
    layer_size_five_hundreds = {{quniform(1, 10, 1)}} * 500


    # # Single autoencoder
    input = Input(shape=(len(train_data[0]),))
    output = input

    for i in range(int(num_layers)):
        output = Dense(int(layer_size_five_hundreds), activation=activation)(output)

    encoded_activation = {{choice(['relu', 'linear'])}}
    output = Dense(2, activation=encoded_activation, name="encoded", kernel_initializer=initializer, bias_initializer='zeros')(output) #maybe just try avoiding dead neurons in small encoding layer?
    
    for i in range(int(num_layers)):
        output = Dense(int(layer_size_five_hundreds), activation=activation)(output)

    output_activation = {{choice(['linear', 'sigmoid'])}}
    output = Dense(len(train_data[0]), activation=output_activation)(output)#'linear',)(output) # First test seems to indicate no change on output with linear

    autoencoder = Model(input, output)

    learning_rate = {{loguniform(math.log(0.1), math.log(0.0001))}}
    optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    autoencoder.compile(
        optimizer=optimizer, #adamax and nadam worked well too
        loss='mean_squared_error',
        metrics=['mse']
    )

    # from keras.callbacks import TensorBoard
    # tensorboard = TensorBoard(log_dir="logs/{}".format(datetime.datetime.now().strftime("%I%M%p%B%d%Y")), write_graph=False)#,histogram_freq=2,)#, histogram_freq=1, write_graph=False)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')
    batch_size_power_of_2 = 2 ** int({{quniform(6, 11, 1)}})

    print('num_layers', num_layers)
    print('layer_size_five_hundreds', layer_size_five_hundreds)
    print('encoded_activation', encoded_activation)
    print('output_activation', output_activation)
    print('learning_rate', learning_rate)
    print('batch_size_power_of_2', batch_size_power_of_2)
    autoencoder.fit(
        train_data, train_data,
        epochs=10,
        batch_size=int(batch_size_power_of_2),
        shuffle=True,
        callbacks=[early_stop],
        validation_data=(test_data, test_data)
    )

    loss, mse = autoencoder.evaluate(test_data, test_data, verbose=0)
    print('Test accuracy:', mse)
    K.clear_session()
    return {'loss': mse, 'status': STATUS_OK, 'model': autoencoder}


def data():
    training_sample_size = 2**20 #~1000000
    test_sample_size = 2**14

    start = time.time()
    print("Generating training data...")
    train_data = pendulums.generate_samples(training_sample_size)
    test_data = pendulums.generate_samples(test_sample_size)
    print("Took this long to load:", time.time() - start)
    # TODO This could be more efficient
    #train_data = generate_box_samples(training_sample_size, x_bounds, y_bounds)
    
    return train_data, test_data

if __name__ == "__main__":
    # training_sample_size = 1000000
    # test_sample_size = 100

    # print("Generating training data...")

    # # TODO This could be more efficient
    # #train_data = generate_box_samples(training_sample_size, x_bounds, y_bounds)
    # train_data = pendulums.generate_samples(training_sample_size)
    # test_data = pendulums.generate_samples(test_sample_size)

    from hyperopt import Trials
    import hyperopt.plotting

    trials = Trials()
    start = time.time()
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          trials=trials,
                                          max_evals=200,
                                          eval_space=True)
    train, test = data()
    print("Evalutation of best performing model:")
    #print(best_model.evaluate(test, test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    class Bandit:
        def __init__(self, space):
            self.params = space

    #hyperopt.plotting.main_plot_history(trials, do_show=True)

    #hyperopt.plotting.main_plot_histogram(trials, do_show=True)
    #hyperopt.plotting.main_plot_vars(trials, Bandit(space), do_show=True)

    import matplotlib.gridspec as gridspec

    n = len(best_run)
    gs = gridspec.GridSpec(n // 2, n // 2 + n % 2)

    fig = plt.figure()
    for key, gs_i in zip(best_run, gs):
        ax = fig.add_subplot(gs_i)
        xs = [t['misc']['vals'][key] for t in trials.trials]
        ys = [-t['result']['loss'] for t in trials.trials]
        ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5)
        ax.set_title(key, fontsize=18)
        ax.set_xlabel(key, fontsize=12)
        ax.set_ylabel('cross validation accuracy', fontsize=12)

    print(trials.trials)
    print("Took:", time.time()-start)
    plt.tight_layout()
    plt.show()