import time
import datetime
import pickle
import numpy

import keras
from keras.layers import Input, Dense
from keras.models import Model, load_model

# import viz

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    start_time = time.time()
    
    training_sample_size = 10000
    test_sample_size = 1000

    print("Loading training data...")
    data = load_data('mass-spring-bar-configurations.pickle')
    
    train_data = numpy.array(data[:training_sample_size])
    test_data = numpy.array(data[training_sample_size:training_sample_size+test_sample_size])

    print("Done loading: ", time.time()-start_time)
    
    # this is the size of our encoded representations
    encoded_dim = 3

    # Single autoencoder
    # initializer = keras.initializers.RandomUniform(minval=0.0, maxval=0.01, seed=5)
    # bias_initializer = initializer
    activation = 'relu' #keras.layers.advanced_activations.LeakyReLU(alpha=0.3) #'relu'
    
    input = Input(shape=(len(train_data[0]),))
    output = Dense(200, activation=activation)(input)
    output = Dense(100, activation=activation)(output)
    output = Dense(encoded_dim, activation=activation, name="encoded")(output)
    output = Dense(100, activation=activation)(output)
    output = Dense(200, activation=activation)(output)
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
        epochs=10,
        batch_size=256,
        shuffle=True,
        validation_data=(test_data, test_data)
    )

    output_path = 'models/' + datetime.datetime.now().strftime("%I %M%p %B %d %Y") + '.h5'
    autoencoder.save(output_path)

    print("Total model time: ", time.time() - model_start_time)

    # Display
    predict_start = time.time()
    test_data = test_data
    decoded_boxes = autoencoder.predict(test_data)
    print('Predict took: ', time.time() - predict_start)
    
    print("Total runtime: ", time.time() - start_time)

    # Draw some output
    # viz.render(q, [])
    # viz.render(q_predict, [], color=viz.red)

if __name__ == "__main__":
    main()
