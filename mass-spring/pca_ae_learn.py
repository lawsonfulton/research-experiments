import time
import datetime
import pickle
import numpy


# import viz

def load_autoencoder(path):
    print("Loading model...")
    autoencoder = load_model(path)
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

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    start_time = time.time()
    
    training_sample_size = 10000
    test_sample_size = 1000

    print("Loading training data...")
    data = numpy.array(load_data('mass-spring-bar-configurations_small.pickle'))
    numpy_base_verts = data[0]
    numpy.random.shuffle(data)
    print(data)
    ### PCA Version
    numpy_displacements_sample = data - numpy_base_verts # should convert this to offsets?

    print("Doing PCA...")
    num_verts = len(data[0]) // 2
    num_samples = training_sample_size
    train_size = num_samples
    test_size = num_samples
    test_data = numpy_displacements_sample[:test_size] * 1.0
    #numpy.random.shuffle(numpy_displacements_sample)

    train_data = numpy_displacements_sample[0:train_size]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    pca.fit(train_data.reshape((train_size, 2 * num_verts)))

    # def encode(q):
    #     return pca.transform(numpy.array([q.flatten() - numpy_base_verts]))[0]

    # def decode(z):
    #     return (numpy_base_verts + pca.inverse_transform(numpy.array([z]))[0]).reshape((num_verts, 3))

    # print(numpy.equal(test_data[0].flatten().reshape((len(test_data[0]),3)), test_data[0]))
    # print(encode(test_data[0]))

    test_data_pca_encoded = pca.transform(test_data.reshape(test_size, 2 * num_verts))
    test_data_pca_decoded = (numpy_base_verts + pca.inverse_transform(test_data_pca_encoded)).reshape(test_size, num_verts, 2)
    ### End of PCA version



    start_time = time.time()
    
    train_size = num_samples
    test_size = num_samples
    test_data = test_data_pca_encoded[:test_size]
    
    # numpy.random.shuffle(test_data_pca_encoded)
    # train_data = numpy_verts_sample[test_size:test_size+train_size]
    train_data = test_data_pca_encoded[0:train_size]

    mean = numpy.mean(train_data, axis=0)
    std = numpy.std(train_data, axis=0)
    
    mean = numpy.mean(train_data)
    std = numpy.std(train_data)

    s_min = numpy.min(train_data)
    s_max = numpy.max(train_data)
    

    def normalize(data):
        return numpy.nan_to_num((data - mean) / std)
        # return numpy.nan_to_num((train_data - s_min) / (s_max - s_min))
    def denormalize(data):
        return data * std + mean
        # return data * (s_max - s_min) + s_min

    train_data = normalize(train_data)
    test_data = normalize(test_data)

    # print(train_data)
    # print(mean)
    # print(std)
    # exit()
    # this is the size of our encoded representations
    encoded_dim = 3

    # Single autoencoder
    # initializer = keras.initializers.RandomUniform(minval=0.0, maxval=0.01, seed=5)
    # bias_initializer = initializer
    import keras
    from keras.layers import Input, Dense
    from keras.models import Model, load_model

    activation = keras.layers.advanced_activations.LeakyReLU(alpha=0.3) #'relu'
    
    input = Input(shape=(len(train_data[0]),))
    output = input
    
    output = Dense(512, activation=activation)(output)
    output = Dense(64, activation=activation)(output)
    output = Dense(encoded_dim, activation=activation, name="encoded")(output)
    output = Dense(64, activation=activation)(output)
    output = Dense(512, activation=activation)(output)
    
    output = Dense(len(train_data[0]), activation='linear')(output)#'linear',)(output) # First test seems to indicate no change on output with linear

    autoencoder = Model(input, output)

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    autoencoder.compile(
        optimizer=optimizer,
        loss='mean_squared_error'
    )
    
    model_start_time = time.time()
    autoencoder.fit(
        train_data, train_data,
        epochs=3000,
        batch_size=num_samples,
        shuffle=True,
        validation_data=(test_data, test_data)
    )

    # output_path = 'trained_models/' + datetime.datetime.now().strftime("%I %M%p %B %d %Y") + '.h5'
    # autoencoder.save(output_path)

    print("Total model time: ", time.time() - model_start_time)

    # Display
    
    ae_decoded_samples = denormalize(autoencoder.predict(test_data))
    #ae_decoded_samples = autoencoder.predict(test_data) * std + mean

    test_data_decoded = (numpy_base_verts + pca.inverse_transform(ae_decoded_samples)).reshape(test_size, num_verts, 2)

    ### Vanilla
    # train_data = numpy.array(data[:training_sample_size])
    # test_data = numpy.array(data[training_sample_size:training_sample_size+test_sample_size])

    # print("Done loading: ", time.time()-start_time)
    
    # # this is the size of our encoded representations
    # encoded_dim = 3

    # # Single autoencoder
    # # initializer = keras.initializers.RandomUniform(minval=0.0, maxval=0.01, seed=5)
    # # bias_initializer = initializer
    # activation = 'linear'#keras.layers.advanced_activations.LeakyReLU(alpha=0.3) #'relu'
    # alpha = 0.3

    # input = Input(shape=(len(train_data[0]),))
    # output = Dense(200, activation=activation)(input)
    # output = keras.layers.advanced_activations.LeakyReLU(alpha=alpha)(output)
    # output = Dense(100, activation=activation)(output)
    # output = keras.layers.advanced_activations.LeakyReLU(alpha=alpha)(output)
    # output = Dense(encoded_dim, activation=activation, name="encoded")(output)
    # output = Dense(100, activation=activation)(output)
    # output = keras.layers.advanced_activations.LeakyReLU(alpha=alpha)(output)
    # output = Dense(200, activation=activation)(output)
    # output = keras.layers.advanced_activations.LeakyReLU(alpha=alpha)(output)
    # output = Dense(len(train_data[0]), activation='linear')(output)#'linear',)(output) # First test seems to indicate no change on output with linear

    # autoencoder = Model(input, output)

    # optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    # autoencoder.compile(
    #     optimizer=optimizer,
    #     loss='mean_squared_error'
    # )
    
    # model_start_time = time.time()
    # autoencoder.fit(
    #     train_data, train_data,
    #     epochs=20,
    #     batch_size=200,#training_sample_size,
    #     shuffle=True,
    #     validation_data=(test_data, test_data)
    # )

    # output_path = 'models/' + datetime.datetime.now().strftime("%I %M%p %B %d %Y") + '.h5'
    # autoencoder.save(output_path)
    ### Vanilla

    print("Total model time: ", time.time() - model_start_time)

    # Display
    predict_start = time.time()
    test_data = test_data
    decoded_samples = autoencoder.predict(test_data)
    print('Predict took: ', time.time() - predict_start)
    
    print("Total runtime: ", time.time() - start_time)

    # Draw some output
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim([0, 0.8])
    ax.set_ylim([0, 0.8])
    ax.set_aspect('equal')

    for i in range(10):
        sample = test_data[i]
        decoded_sample = decoded_samples[i]
        plt.scatter(sample[0::2], sample[1::2], c='b', s=3)
        plt.scatter(decoded_sample[0::2], decoded_sample[1::2], c='r', s=2)
    plt.show()
if __name__ == "__main__":
    main()
