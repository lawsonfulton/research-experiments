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
#import keras_autoencoder as autoencoder

def add_noise(samples):
    return samples #+ np.random.normal(0.0, 0.005, (len(samples), len(samples[0])))

def custom_loss(model):
    def contractive_loss(y_pred, y_true):
        lam = 1e-3
        mse = K.mean(K.square(y_true - y_pred), axis=1)

        W = K.variable(value=model.get_layer('encoded').get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = model.get_layer('encoded').output
        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)

        return mse + contractive

    return contractive_loss

def load_autoencoder(path=None, model=None):
    assert (path is None) != (model is None)

    if path is not None:
        print("Loading model...")
        autoencoder = load_model(path, custom_objects={'contractive_loss': lambda x, y: K.mean(y-x)}) # TODO, fix for custom loss
        print("Done.")
    else:
        autoencoder = model

    def get_encoded_layer_and_index(): # Stupid hack
        for i, layer in enumerate(autoencoder.layers):
            if layer.name == 'encoded':
                return layer, i

    encoded_layer, encoded_layer_idx = get_encoded_layer_and_index()
    encoder = Model(inputs=autoencoder.input, outputs=encoded_layer.output)

    decoder_input = Input(shape=(encoded_layer.output_shape[-1],))
    old_decoder_layers = autoencoder.layers[encoded_layer_idx+1:] # Need to rebuild the tensor I guess
    decoder_output = decoder_input
    for layer in old_decoder_layers:
        decoder_output = layer(decoder_output)

    decoder = Model(inputs=decoder_input, outputs=decoder_output)

    return autoencoder, encoder, decoder

def animate(autoencoder, encoder, decoder):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import animation

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1,2,1)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_aspect('equal')

    ax3d = fig.add_subplot(1,2,2, projection='3d')
    ax3d.set_xlim([0,1.0])
    ax3d.set_ylim([0,1.0])
    ax3d.set_zlim([0,1.0])

    n_samples = 100
    r = 6
    thetas = np.linspace(0.0, 8*math.pi, num=n_samples) #np.zeros(n_samples)
    offsets = np.array([[r * math.sin(theta), r * math.cos(theta)] for theta in np.linspace(0.0, 2*math.pi, num=n_samples)])

    real_pendulums = pendulums.generate_samples(offsets=offsets, thetas=thetas)
    encoded_pendulums = encoder.predict(real_pendulums)
    decoded_pendulums = decoder.predict(encoded_pendulums)

    line, = ax3d.plot(encoded_pendulums[0:1,0], encoded_pendulums[0:1,1], encoded_pendulums[0:1,2])
    def animate(i):
    #     line.set_ydata(np.sin(x + i/10.0))  # update the data
        ax.clear()
        pendulums.draw_from_samples(ax, [real_pendulums[i]], 'r', linewidth=5)
        pendulums.draw_from_samples(ax, [decoded_pendulums[i]], 'b')
        
        line.set_data(encoded_pendulums[:i,0],encoded_pendulums[:i,1])
        line.set_3d_properties(encoded_pendulums[:i, 2])
    print("animating")

    anim = animation.FuncAnimation(fig, animate, frames=n_samples, interval=1000/25, blit=False)#True)
    print("loading video")
    #anim.to_html5_video()
    #anim.save(output_path, writer='imagemagick')
    print("done")
    return anim

def jacobian_output_wrt_input(model): # TODO n = batch_size
    print("Computing jacobian function...")
    n = 1 # Runs on single data point for now

    qs = model.input #qs
    xs = model.output#xs

    dxsdqs = [tf.gradients(tf.slice(xs,[0,i],[n,1]), qs) for i in range(model.output_shape[-1])] #use tf unpack instead?
    jacobian_x_wrt_q = tf.stack(dxsdqs)
    print("Done.")

    #tf_sess = K.get_session()
    tf_sess = tf.Session()
    tf_sess.run(tf.initialize_all_variables())

    def eval_jac(q):
        input_points = np.array([q])
        evaluated_gradients = tf_sess.run(jacobian_x_wrt_q, feed_dict={qs: input_points})
        return np.stack([evaluated_gradients[i][0][0] for i in range(len(evaluated_gradients))])

    return eval_jac

def eval_jacobian(jacobian, qs):
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    return sess.run(jacobian, feed_dict={qs:qs})

def explore_model(decoder):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_aspect('equal')

    predicted_pendulums = decoder.predict(np.array([[j/10.0,0, 0] for i in range(10) for j in range(10)]))
    #print(predicted_pendulums)
    pendulums.draw_from_samples(ax, predicted_pendulums, 'b')
    plt.show()
    print("Done.")

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
    training_sample_size = 5000000
    test_sample_size = 100

    print("Generating training data...")

    # TODO This could be more efficient
    #train_data = generate_box_samples(training_sample_size, x_bounds, y_bounds)
    train_data = pendulums.generate_samples(training_sample_size)
    test_data = pendulums.generate_samples(test_sample_size, x_bounds, y_bounds)
    # TODO test data should be from a different part of the plane to make sure we are generalizing
    print("Done. Runtime: ", time.time()-start_time)
    
    model_start_time = time.time()
    

    # Needed for relu but apparently not for elu
    # For some reason I can't learn high frequency functions with relu alone, and the positive initializer seems
    # to mess with elu
    # initializer = 'glorot_uniform'
    # activation = 'elu'

    initializer = keras.initializers.RandomUniform(minval=0.0001, maxval=0.01)
    # bias_initializer = initializer
    activation = 'linear' #keras.layers.advanced_activations.LeakyReLU(alpha=0.3) #'relu'

    from keras.layers.advanced_activations import LeakyReLU
    alpha = 0.001

    # # Single autoencoder
    input = Input(shape=(len(train_data[0]),))
    output = input
    output = Dense(100, activation=activation)(output)
    output = LeakyReLU(alpha=alpha)(output)
    output = Dense(100, activation=activation)(output)
    output = LeakyReLU(alpha=alpha)(output)
    output = Dense(2, activation=activation, name="encoded", kernel_initializer=initializer, bias_initializer='zeros')(output) #maybe just try avoiding dead neurons in small encoding layer?
    output = LeakyReLU(alpha=alpha)(output)
    output = Dense(100, activation=activation)(output)
    output = LeakyReLU(alpha=alpha)(output)
    output = Dense(100, activation=activation)(output)
    output = LeakyReLU(alpha=alpha)(output)
    output = Dense(len(train_data[0]), activation='linear')(output)#'linear',)(output) # First test seems to indicate no change on output with linear

    autoencoder = Model(input, output)

    # stacked? autoencoder
    # main_input = Input(shape=(len(train_data[0]),))
    # output1 = main_input
    # output2 = Dense(100, activation=activation)(output1)
    # output3 = Dense(3, activation=activation, name="encoded1", kernel_initializer=initializer, bias_initializer='zeros')(output2)
    # output4 = Dense(100, activation=activation)(output3)
    # aux_output = Dense(len(train_data[0]), activation='linear', name="auxout")(output4)#'linear',)(output) # First test seems to indicate no change on output with linear

    # output6 = Dense(2, activation=activation, name="encoded2", kernel_initializer=initializer, bias_initializer='zeros')(output3) #maybe just try avoiding dead neurons in small encoding layer?
    # output7 = Dense(100, activation=activation)(output6)
    # main_output = Dense(len(train_data[0]), activation='linear', name="mainout")(output7)#'linear',)(output) # First test seems to indicate no change on output with linear

    # autoencoder = Model(inputs=[main_input], outputs=[main_output, aux_output])

    # optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    # autoencoder.compile(
    #     optimizer=optimizer,
    #     loss='mean_squared_error' #custom_loss(autoencoder)#
    # )

    # start = time.time()
    # autoencoder.fit(
    #     [train_data], [train_data,train_data],
    #     epochs=10,
    #     batch_size=4096,
    #     shuffle=True,
    #     #callbacks=[OnEpochEnd()],
    #     validation_data=([test_data], [test_data,test_data])
    # )

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    autoencoder.compile(
        optimizer= optimizer, #adamax and nadam worked well too
        loss='mean_squared_error' #custom_loss(autoencoder)#
    )

    start = time.time()
    autoencoder.fit(
        add_noise(train_data), train_data,
        epochs=300,
        batch_size=4096,
        shuffle=True,
        #callbacks=[OnEpochEnd()],
        validation_data=(test_data, test_data)
    )

    output_path = 'models/' + datetime.datetime.now().strftime("%I %M%p %B %d %Y") + '.h5'
    autoencoder.save(output_path)

    ### MY stacked autoencoder
    print("Total model time: ", time.time() - model_start_time)

    #show
    # encode and decode some digits
    # note that we take them from the *test* set
    predict_start = time.time()
    test_data = add_noise(test_data)
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
