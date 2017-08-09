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

import boxes
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

def load_autoencoder(path):
    print("Loading model...")
    autoencoder = load_model(path, custom_objects={'contractive_loss': lambda x, y: K.mean(y-x)}) # TODO, fix for custom loss
    print("Done.")

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

def animate(autoencoder, encoder, decoder, output_path='autoencoder.gif'):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import animation

    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1,2,1)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_aspect('equal')

    ax3d = fig.add_subplot(1,2,2, projection='3d')
    ax3d.set_xlim([0.2,1.0])
    ax3d.set_ylim([0.2,1.0])
    ax3d.set_zlim([0,1.0])

    n_samples = 100
    r = 6
    thetas = np.linspace(0.0, 8*math.pi, num=n_samples) #np.zeros(n_samples)
    offsets = np.array([[r * math.sin(theta), r * math.cos(theta)] for theta in np.linspace(0.0, 2*math.pi, num=n_samples)])

    real_boxes = boxes.generate_samples(offsets=offsets, thetas=thetas)
    encoded_boxes = encoder.predict(real_boxes)
    decoded_boxes = decoder.predict(encoded_boxes)

    line, = ax3d.plot(encoded_boxes[0:1,0], encoded_boxes[0:1,1], encoded_boxes[0:1,2])
    def animate(i):
    #     line.set_ydata(np.sin(x + i/10.0))  # update the data
        ax.clear()
        boxes.draw_from_samples(ax, [real_boxes[i]], 'r', linewidth=5)
        boxes.draw_from_samples(ax, [decoded_boxes[i]], 'b')
        
        line.set_data(encoded_boxes[:i,0],encoded_boxes[:i,1])
        line.set_3d_properties(encoded_boxes[:i, 2])
    print("animating")

    anim = animation.FuncAnimation(fig, animate, frames=n_samples, interval=10, blit=False)#True)
    print("loading video")
    #anim.to_html5_video()
    anim.save(output_path, writer='imagemagick')
    print("done")

def jacobian_output_wrt_input(model): # TODO n = batch_size
    print("Computing jacobian function...")
    n = 1 # Runs on single data point for now

    xs = decoder.input #qs
    ys = decoder.output#xs

    dysdxs = [tf.gradients(tf.slice(ys,[0,i],[n,1]), xs) for i in range(model.output_shape[-1])] #use tf unpack instead?
    jacobian_y_wrt_x = tf.stack(dysdxs)
        
    print("Done.")
    return jacobian_y_wrt_x

def eval_jacobian(jacobian, xs):
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    return sess.run(jacobian_x_wrt_q, feed_dict={xs:xs})

def explore_model(decoder):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_aspect('equal')

    predicted_boxes = decoder.predict(np.array([[j/10.0,0, 0] for i in range(10) for j in range(10)]))
    #print(predicted_boxes)
    boxes.draw_from_samples(ax, predicted_boxes, 'b')
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
    box_dim = 8

    # Needed for relu but apparently not for elu
    # For some reason I can't learn high frequency functions with relu alone, and the positive initializer seems
    # to mess with elu
    # initializer = 'glorot_uniform'
    # activation = 'elu'

    # initializer = keras.initializers.RandomUniform(minval=0.0, maxval=0.01, seed=5)
    # bias_initializer = initializer
    activation = 'relu' #keras.layers.advanced_activations.LeakyReLU(alpha=0.3) #'relu'

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

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    autoencoder.compile(
        optimizer=optimizer,
        loss=custom_loss(autoencoder)#'mean_squared_error'
    )

    start = time.time()

    class OnEpochEnd(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            decoded_boxes = autoencoder.predict(test_data)

            # Draw some output
            num_to_show = 15
            boxes.draw_from_samples(ax, test_data[:num_to_show], 'r', linewidth=5)
            boxes.draw_from_samples(ax, decoded_boxes[:num_to_show], 'b',linewidth=2)

            from IPython.display import display
            display(fig)
            ax.clear()
            # fig.show()
    print("update")
    autoencoder.fit(
        add_noise(train_data), train_data,
        epochs=250  ,
        batch_size=8192,
        shuffle=True,
        #callbacks=[OnEpochEnd()],
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
