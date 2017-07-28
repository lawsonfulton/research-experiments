import keras
from keras.layers import Input, Dense
from keras.models import Model, Sequential, load_model
from keras import regularizers

def get_autoencoder(path):
    print("Loading...")
    model = load_model(path)
    decoder_input = Input(shape=(3,))
    decoder_middle = model.layers[2](decoder_input)
    decoded = model.layers[3](decoder_middle)
    decoder = Model(decoder_input, decoded)

    encoder_input = Input(shape=(8,))
    encoder_middle = model.layers[0](encoder_input)
    encoded = model.layers[1](encoder_middle)
    encoder = Model(encoder_input, encoded)
    print("Done.")
    
    return encoder, decoder

def train_model(train_data, test_data, layer_dims, learning_rate=0.01, epochs=10, batch_size=512,loss='mean_squared_logarithmic_error', saved_model_path=None):
    input_dim = layer_dims[0]
    encoding_dim = layer_dims[-1]
    
    kernel_initializer = keras.initializers.RandomUniform(minval=0.0, maxval=0.05, seed=None)
    activation = 'relu'

    new_model = True
    if new_model:
        encoder_input = Input(shape=(input_dim,))
        prev_input = encoder_input
        for dim in layer_dims[1:]:
            encoded = Dense(dim, activation=activation, kernel_initializer=kernel_initializer)(prev_input)
            prev_input = encoded

        encoder = Model(encoder_input, encoded)

        decoder_input = Input(shape=(encoding_dim,))
        prev_input = decoder_input
        for dim in reversed(layer_dims[0:-1]):
            decoded = Dense(dim, activation=activation, kernel_initializer=kernel_initializer)(prev_input)
            prev_input = decoded

        decoder = Model(decoder_input, decoded)


        autoencoder_input = Input(shape=(input_dim,))
        encoded = encoder(autoencoder_input)
        decoded = decoder(encoded)

        autoencoder = Model(autoencoder_input, decoded)
    else:    
        autoencoder = Sequential()
        autoencoder.add(Dense(layer_dims[1], input_shape=(input_dim,), activation=activation, kernel_initializer=kernel_initializer, name='encoder'))
        #Encoding layers
        for layer_dim in layer_dims[2:]:
            autoencoder.add(Dense(layer_dim, activation=activation, kernel_initializer=kernel_initializer))
        
        #Decoding layers
        for layer_dim in reversed(layer_dims[0:-1]):
            autoencoder.add(Dense(layer_dim, activation=activation, kernel_initializer=kernel_initializer))

    #TODO should output layer be linear??

    optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(optimizer=optimizer, loss=loss)

    #train
    autoencoder.fit(train_data, train_data,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(test_data, test_data))

    if saved_model_path:
        autoencoder.save(saved_model_path)

    return autoencoder
