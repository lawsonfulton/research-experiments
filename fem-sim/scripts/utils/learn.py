import time
import subprocess
import json
import os

import numpy
import scipy
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import pyigl as igl
from utils.iglhelpers import e2p, p2e
from utils import my_utils

# Global Vars
modes = {0: 'baseline', 1:'pca', 2:'autoencoder'}
current_mode = 0
current_frame = 0
visualize_test_data = True
# training_data_root = 'training_data/first_interaction/'
# test_data_root = 'training_data/test_interaction/'
training_data_root = 'training_data/fixed_material_model/'
test_data_root = 'training_data/fixed_material_model_test/'

def autoencoder_analysis(
    data,
    test_data,
    activation='relu',
    latent_dim=3,
    epochs=100,
    batch_size=100,
    layers=[32, 16],
    pca_weights=None,
    pca_basis=None,
    do_fine_tuning=False,
    model_root=None,
    autoencoder_config=None,
    callback=None,
    ): # TODO should probably just pass in both configs here 
    """
    Returns and encoder and decoder for going into and out of the reduced latent space.
    If pca_weights is given, then do a weighted mse.
    """
    assert not((pca_weights is not None) and (pca_basis is not None))  # pca_weights incompatible with pca_object

    import keras
    import keras.backend as K
    from keras.layers import Input, Dense, Lambda
    from keras.models import Model, load_model
    from keras.engine.topology import Layer
    from keras.callbacks import History 

    flatten_data, unflatten_data = my_utils.get_flattners(data)

    # TODO: Do I need to shuffle?
    train_data = data
    test_data = test_data if test_data is not None else data[:10] 


    ## Preprocess the data
    # mean = numpy.mean(train_data, axis=0)
    # std = numpy.std(train_data, axis=0)
    
    mean = numpy.mean(train_data)
    std = numpy.std(train_data)
    s_min = numpy.min(train_data)
    s_max = numpy.max(train_data)
    print(mean)
    print(std)

    # TODO dig into this. Why does it mess up?
    def normalize(data):
        return data
        #return (data - mean) / std
        # return numpy.nan_to_num((train_data - s_min) / (s_max - s_min))
    def denormalize(data):
        return data
        #return data * std + mean
        # return data * (s_max - s_min) + s_mi

    ## Set up the network
    
    input = Input(shape=(len(train_data[0]),), name="encoder_input")
    output = input
    
    if pca_basis is not None:
        # output = Lambda(pca_transform_layer)(output)
        # output = PCALayer(pca_object, is_inv=False, fine_tune=do_fine_tuning)(output)

        W = pca_basis
        b = numpy.zeros(pca_basis.shape[1])
        output = Dense(pca_basis.shape[1], activation='linear', weights=[W,b], trainable=do_fine_tuning, name="pca_encode_layer")(output)

    for i, layer_width in enumerate(layers):
        act = activation
        # if i == len(layers) - 1:
        #     act = 'sigmoid'
        output = Dense(layer_width, activation=act, name="dense_encode_layer_" + str(i))(output)

    # -- Encoded layer
    output = Dense(latent_dim, activation=activation, name="encoded_layer")(output)  # TODO Tanh into encoded layer to bound vars?

    for i, layer_width in enumerate(reversed(layers)):
        output = Dense(layer_width, activation=activation, name="dense_decode_layer_" + str(i))(output)
    
    if pca_basis is not None:
        output = Dense(pca_basis.shape[1], activation='linear', name="to_pca_decode_layer")(output) ## TODO is this right?? Possibly should just change earlier layer width
        # output = Lambda(pca_inv_transform_layer)(output)
        # output = PCALayer(pca_object, is_inv=True, fine_tune=do_fine_tuning)(output)

        W = pca_basis.T
        b = numpy.zeros(len(train_data[0]))
        output = Dense(len(train_data[0]), activation='linear', weights=[W,b], trainable=do_fine_tuning, name="pca_decode_layer")(output)
    else:
        output = Dense(len(train_data[0]), activation='linear', name="decoder_output_layer")(output)#'linear',)(output) # First test seems to indicate no change on output with linear

    autoencoder = Model(input, output)
    ## Set the optimization parameters

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    autoencoder.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
    )

    hist = History()
    model_start_time = time.time()

    autoencoder.fit(
        train_data, train_data,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(test_data, test_data),
        callbacks=[hist, callback]
        )

    training_time = time.time() - model_start_time
    print("Total training time: ", training_time)
    
    autoencoder,encoder, decoder = my_utils.decompose_ae(autoencoder, do_energy=False)

    # Save the models in both tensorflow and keras formats
    if model_root:
        print("Saving model...")
        models_with_names = [
            (autoencoder, "autoencoder"),
            (encoder, "encoder"),
            (decoder, "decoder"),
        ]

        keras_models_dir = os.path.join(model_root, 'keras_models')
        my_utils.create_dir_if_not_exist(keras_models_dir)

        for keras_model, name in models_with_names:
            keras_model_file = os.path.join(keras_models_dir, name + ".hdf5")
            keras_model.save(keras_model_file)  # Save the keras model

        print ("Finished saving model.")

    def encode(decoded_data):
        return encoder.predict(normalize(flatten_data(decoded_data))) 
    def decode(encoded_data):
        return unflatten_data(denormalize(decoder.predict(encoded_data)))

    return encode, decode, training_time

def pca_analysis(data, n_components, plot_explained_variance=False):
    """ Returns and encoder and decoder for projecting into and out of the PCA reduced space """
    print("Doing PCA with", n_components, "components...")
    # numpy.random.shuffle(numpy_displacements_sample) # Necessary?
    pca = PCA(n_components=n_components)

    flatten_data, unflatten_data = my_utils.get_flattners(data)

    flattened_data = flatten_data(data)
    pca.fit(flattened_data)

    if plot_explained_variance:
        explained_var = pca.explained_variance_ratio_
        plt.xlabel('Component')
        plt.ylabel('Explained Variance')
        plt.plot(list(range(1, n_components + 1)), explained_var)
        plt.show(block=False)

        print("Total explained variance =", sum(explained_var))

    mse = mean_squared_error(flattened_data, pca.inverse_transform(pca.transform(flattened_data)))

    def encode(decoded_data):
        #return pca.transform(flatten_data(decoded_data))        
        #return pca.components_ @ flatten_data(decoded_data).T # We don't need to subtract the mean before going to/from reduced space?
        X = flatten_data(decoded_data) - pca.mean_
        return numpy.dot(X, pca.components_.T)

    def decode(encoded_data):
        # return unflatten_data(pca.inverse_transform(encoded_data))
        # return unflatten_data((pca.components_.T @ encoded_data).T)
        X_original = numpy.dot(encoded_data, pca.components_)
        X_original = X_original + pca.mean_
        return unflatten_data(X_original)

    return pca, encode, decode, pca.explained_variance_ratio_, mse

def pca_no_centering(samples, pca_dim):
    from scipy import linalg
    from sklearn.utils.extmath import svd_flip

    print('Doing PCA for', pca_dim, 'components...')

    _, S, components = linalg.svd(samples, full_matrices=False)
    
    U = components[:pca_dim].T

    explained_variance_ = (S ** 2) / (len(samples) - 1)
    total_var = explained_variance_.sum()
    explained_variance_ratio = (explained_variance_ / total_var)[:pca_dim]

    def encode(samples):
        return samples @ U

    def decode(samples):
        return samples @ U.T

    return U, explained_variance_ratio, encode, decode

def discrete_nn_analysis(
    reduced_space_samples,
    energy_samples,
    reduced_space_validation_samples,
    energy_validation_samples,
    volumes,
    activation='relu',
    epochs=100,
    batch_size=100,
    layers=[32, 16],
    do_fine_tuning=False,
    model_root=None,
    autoencoder_config=None,
    energy_model_config=None): # TODO should probably just pass in both configs here 
    """
    Returns and encoder and decoder for going into and out of the reduced latent space.
    If pca_weights is given, then do a weighted mse.
    If pca_object is given, then the first and final layers will do a pca transformation of the reduced_space_samples.
    """
    

    import keras
    import keras.backend as K
    from keras.layers import Input, Dense, Lambda
    from keras.models import Model, load_model
    from keras.engine.topology import Layer
    from keras.callbacks import History 

    flatten_data, unflatten_data = my_utils.get_flattners(reduced_space_samples)

    # TODO: Do I need to shuffle?
    reduced_space_samples = flatten_data(reduced_space_samples)
    energy_samples = flatten_data(energy_samples)
    reduced_space_validation_samples = flatten_data(reduced_space_samples[:10] if reduced_space_validation_samples is None else reduced_space_validation_samples)
    energy_validation_samples = flatten_data(energy_samples[:10] if energy_validation_samples is None else energy_validation_samples)


    input = Input(shape=(len(reduced_space_samples[0]),), name="energy_model_input")
    output = input

    for i, layer_width in enumerate(layers):
        output = Dense(layer_width, activation=activation, name="dense_decode_layer_" + str(i))(output)

    def lp_reg(p, lam):
        def reg(vec):
            # return lam * K.pow(K.sum(K.pow(K.abs(vec), p)), 1.0 / p)
            return lam * K.sum(K.pow(K.abs(vec), p))
        return reg

    def l1_volume_reg(lam):
        vol = K.variable(volumes)
        def reg(vec):
            return lam * K.sum(K.abs(vec) * vol)
            # return lam * K.sum(K.abs(vec))

        return reg

    output = Dense(len(energy_samples[0]), activation='relu', name="output_layer" + str(i),
                    # activity_regularizer=keras.regularizers.l1(0.0009), # TODO what val? default was 0.01
                    # activity_regularizer=lp_reg(0.05, 0.005), # TODO what val? default was 0.01
                    activity_regularizer=l1_volume_reg(0.001)
                    )(output)

    model = Model(input, output)

    def energy_loss(y_true, y_pred):
        return K.mean(K.square(K.sum(y_pred * y_true, axis=-1) - K.sum(y_true, axis=-1))) # TODO should mean be over an axis?

    def energy_loss_numpy(y_true, y_pred):
        return numpy.mean(numpy.square(numpy.sum(y_pred * y_true, axis=-1) - numpy.sum(y_true, axis=-1)))

    optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(
        optimizer=optimizer,
        loss=energy_loss#'mean_squared_error',
    )
    
    
    hist = History()
    model_start_time = time.time()
    from keras.callbacks import Callback 
    idx = numpy.random.randint(len(reduced_space_samples), size=10)
    class MyHistoryCallback(Callback):
        """Callback that records events into a `History` object.
        This callback is automatically applied to
        every Keras model. The `History` object
        gets returned by the `fit` method of models.
        """

        def on_train_begin(self, logs=None):
            self.epoch = []
            self.history = {}

        def on_epoch_end(self, epoch, logs=None):
            actual = energy_samples[idx, :]
            pred = self.model.predict(reduced_space_samples[idx, :])
            mse = energy_loss_numpy(actual, pred)
            print("Actual energy: ", (actual).sum(axis=1))
            print("Predicted energy: ", numpy.sum(pred * actual, axis=-1))
            nonzeros = [p[numpy.nonzero(p)] for p in pred]
            # print("non zero weights: ", nonzeros)
            print("len(nonzero):", [len(nz) for nz in nonzeros])
            #val_mse = mean_squared_error(high_dim_pca_decode(self.model.predict(encoded_high_dim_pca_test_displacements)), test_displacements)
            self.history.setdefault('mse', []).append(mse)
            # self.history.setdefault('val_mse', []).append(val_mse)
            print()
            print("Mean squared error: ", mse)
            # print("Val Mean squared error: ", val_mse)

            logs = logs or {}
            self.epoch.append(epoch)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

    my_hist_callback = MyHistoryCallback()

    model.fit(
        reduced_space_samples, energy_samples,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        # validation_data=(reduced_space_validation_samples, energy_validation_samples),
        callbacks=[my_hist_callback]
        )

    if model_root:
        print("Saving model...")
        models_with_names = [
            (model, "l1_discrete_energy_model"),
        ]

        keras_models_dir = os.path.join(model_root, 'keras_models')
        my_utils.create_dir_if_not_exist(keras_models_dir)

        for keras_model, name in models_with_names:
            keras_model_file = os.path.join(keras_models_dir, name + ".hdf5")
            keras_model.save(keras_model_file)  # Save the keras model

        print ("Finished saving model.")

    print("Total training time: ", time.time() - model_start_time)
    # model.save('discrete_energy_model.hdf5')
    # if energy_model_confiredict(normalize(flatten_data(decoded_data))) 
    def decode(encoded_data):
        return unflatten_data(model.predict(encoded_data))

    return decode, hist

def binary_discrete_nn_analysis(
    reduced_space_samples,
    energy_samples,
    reduced_space_validation_samples,
    energy_validation_samples,
    activation='relu',
    epochs=100,
    batch_size=100,
    layers=[32, 16],
    do_fine_tuning=False,
    model_root=None,
    autoencoder_config=None,
    energy_model_config=None): # TODO should probably just pass in both configs here 
    """
    Returns and encoder and decoder for going into and out of the reduced latent space.
    If pca_weights is given, then do a weighted mse.
    If pca_object is given, then the first and final layers will do a pca transformation of the reduced_space_samples.
    """
    

    import keras
    import keras.backend as K
    from keras.layers import Input, Dense, Lambda, multiply, ActivityRegularization
    from keras.models import Model, load_model
    from keras.engine.topology import Layer
    from keras.callbacks import History 

    flatten_data, unflatten_data = my_utils.get_flattners(reduced_space_samples)

    # TODO: Do I need to shuffle?
    reduced_space_samples = flatten_data(reduced_space_samples)
    energy_samples = flatten_data(energy_samples)
    reduced_space_validation_samples = flatten_data(reduced_space_samples[:10] if reduced_space_validation_samples is None else reduced_space_validation_samples)
    energy_validation_samples = flatten_data(energy_samples[:10] if energy_validation_samples is None else energy_validation_samples)


    input = Input(shape=(len(reduced_space_samples[0]),), name="energy_model_input")
    output = input

    for i, layer_width in enumerate(layers):
        output = Dense(layer_width, activation=activation, name="dense_decode_layer_" + str(i))(output)

    
    
    fixed_input = Input(tensor=K.variable([[1.0]] ))
    weights = Dense(len(energy_samples[0]), activation=None, use_bias=False)(fixed_input)
    
    # hard_sigmoid?
    output = Dense(len(energy_samples[0]), activation='hard_sigmoid', name="output_layer" + str(i),
                    # activity_regularizer=keras.regularizers.l1(0.0001), # TODO what val? default was 0.01
                    )(output)

    output = multiply([output, weights])
    output = ActivityRegularization(l1=0.001, l2=0.0)(output)

    model = Model([input, fixed_input], output)

    def energy_loss(y_true, y_pred):
        return K.mean(K.square(K.sum(y_pred * y_true, axis=-1) - K.sum(y_true, axis=-1))) # TODO should mean be over an axis?

    def energy_loss_numpy(y_true, y_pred):
        return numpy.mean(numpy.square(numpy.sum(y_pred * y_true, axis=-1) - numpy.sum(y_true, axis=-1)))

    optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(
        optimizer=optimizer,
        loss=energy_loss#'mean_squared_error',
    )
    
    
    hist = History()
    model_start_time = time.time()
    from keras.callbacks import Callback 
    idx = numpy.random.randint(len(reduced_space_samples), size=10)
    class MyHistoryCallback(Callback):
        """Callback that records events into a `History` object.
        This callback is automatically applied to
        every Keras model. The `History` object
        gets returned by the `fit` method of models.
        """

        def on_train_begin(self, logs=None):
            self.epoch = []
            self.history = {}

        def on_epoch_end(self, epoch, logs=None):
            actual = energy_samples[idx, :]
            pred = self.model.predict(reduced_space_samples[idx, :])
            mse = energy_loss_numpy(actual, pred)
            print("Actual energy: ", (actual).sum(axis=1))
            print("Predicted energy: ", numpy.sum(pred * actual, axis=-1))
            nonzeros = [p[numpy.nonzero(p)] for p in pred]
            # print("non zero weights: ", nonzeros)
            print("len(nonzero):", [len(nz) for nz in nonzeros])
            #val_mse = mean_squared_error(high_dim_pca_decode(self.model.predict(encoded_high_dim_pca_test_displacements)), test_displacements)
            self.history.setdefault('mse', []).append(mse)
            # self.history.setdefault('val_mse', []).append(val_mse)
            print()
            print("Mean squared error: ", mse)
            # print("Val Mean squared error: ", val_mse)

            logs = logs or {}
            self.epoch.append(epoch)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

    my_hist_callback = MyHistoryCallback()

    model.fit(
        reduced_space_samples, energy_samples,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        # validation_data=(reduced_space_validation_samples, energy_validation_samples),
        callbacks=[my_hist_callback]
        )

    if model_root:
        print("Saving model...")
        models_with_names = [
            (model, "l1_discrete_energy_model"),
        ]

        keras_models_dir = os.path.join(model_root, 'keras_models')
        my_utils.create_dir_if_not_exist(keras_models_dir)

        for keras_model, name in models_with_names:
            keras_model_file = os.path.join(keras_models_dir, name + ".hdf5")
            keras_model.save(keras_model_file)  # Save the keras model

        print ("Finished saving model.")

    print("Total training time: ", time.time() - model_start_time)
    # model.save('discrete_energy_model.hdf5')
    # if energy_model_confiredict(normalize(flatten_data(decoded_data))) 
    def decode(encoded_data):
        return unflatten_data(model.predict(encoded_data))

    return decode, hist


def fit_nn_direct_energy(
    reduced_space_samples,
    energy_samples,
    reduced_space_validation_samples,
    energy_validation_samples,
    activation='relu',
    epochs=100,
    batch_size=100,
    layers=[32, 16],
    do_fine_tuning=False,
    model_root=None,
    autoencoder_config=None,
    energy_model_config=None): # TODO should probably just pass in both configs here 
    """
    Returns and encoder and decoder for going into and out of the reduced latent space.
    If pca_weights is given, then do a weighted mse.
    If pca_object is given, then the first and final layers will do a pca transformation of the reduced_space_samples.
    """
    

    import keras
    import keras.backend as K
    from keras.layers import Input, Dense, Lambda
    from keras.models import Model, load_model
    from keras.engine.topology import Layer
    from keras.callbacks import History 

    flatten_data, unflatten_data = my_utils.get_flattners(reduced_space_samples)

    # TODO: Do I need to shuffle?
    reduced_space_samples = flatten_data(reduced_space_samples)


    input = Input(shape=(len(reduced_space_samples[0]),), name="energy_model_input")
    output = input

    for i, layer_width in enumerate(layers):
        output = Dense(layer_width, activation=activation, name="dense_decode_layer_" + str(i))(output)

    output = Dense(len(energy_samples[0]), activation='relu', name="output_layer" + str(i),
                    )(output)

    model = Model(input, output)


    optimizer = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(
        optimizer=optimizer,
        loss='mean_absolute_percentage_error',
    )
    
    
    def mean_absolute_percentage_error(y_true, y_pred): 

        ## Note: does not handle mix 1d representation
        #if _is_1d(y_true): 
        #    y_true, y_pred = _check_1d_array(y_true, y_pred)

        return numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100

    hist = History()
    model_start_time = time.time()
    from keras.callbacks import Callback 
    idx = numpy.random.randint(len(reduced_space_samples), size=10)
    class MyHistoryCallback(Callback):
        """Callback that records events into a `History` object.
        This callback is automatically applied to
        every Keras model. The `History` object
        gets returned by the `fit` method of models.
        """

        def on_train_begin(self, logs=None):
            self.epoch = []
            self.history = {}

        def on_epoch_end(self, epoch, logs=None):
            actual = energy_samples[idx, :]
            pred = self.model.predict(reduced_space_samples[idx, :])
            mse = mean_absolute_percentage_error(actual, pred)
            print("Actual energy: ", actual)
            print("Predicted energy: ", pred)
            
            self.history.setdefault('mse', []).append(mse)
            # self.history.setdefault('val_mse', []).append(val_mse)
            print()
            print("Percentage error: ", mse)
            # print("Val Mean squared error: ", val_mse)

            logs = logs or {}
            self.epoch.append(epoch)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

    my_hist_callback = MyHistoryCallback()

    model.fit(
        reduced_space_samples, energy_samples,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        # validation_data=(reduced_space_validation_samples, energy_validation_samples),
        callbacks=[my_hist_callback]
        )

    if model_root:
        print("Saving model...")
        models_with_names = [
            (model, "direct_energy_model"),
        ]

        keras_models_dir = os.path.join(model_root, 'keras_models')
        my_utils.create_dir_if_not_exist(keras_models_dir)

        for keras_model, name in models_with_names:
            keras_model_file = os.path.join(keras_models_dir, name + ".hdf5")
            keras_model.save(keras_model_file)  # Save the keras model

        print ("Finished saving model.")

    print("Total training time: ", time.time() - model_start_time)
    # model.save('discrete_energy_model.hdf5')
    # if energy_model_confiredict(normalize(flatten_data(decoded_data))) 
    def decode(encoded_data):
        return unflatten_data(model.predict(encoded_data))

    return decode, hist

def load_displacements_and_energy(model_root, use_reencoded=False, use_extra_samples=False):
    from keras.models import Model, load_model
    encoder = load_model(os.path.join(model_root,'keras_models/encoder.hdf5'))
    U = igl.eigen.MatrixXd()
    igl.readDMAT(os.path.join(model_root, 'pca_results/ae_pca_components.dmat'), U)

    if not use_reencoded and not use_extra_samples:
        training_data_path = os.path.join(model_root,'training_data/training')
        displacements = my_utils.load_displacement_dmats_to_numpy(training_data_path)
        flatten_displ, unflatten_displ = my_utils.get_flattners(displacements)
        encoded_displacements = encoder.predict(flatten_displ(displacements) @ U)
        energies = my_utils.load_energy_dmats_to_numpy(training_data_path)

    if use_reencoded:
        reencoded_training_data_path = os.path.join(model_root, 'augmented_training_data/reencoded/')
        displacements_path = os.path.join(reencoded_training_data_path, 'displacements.dmat')
        enc_displacements_path = os.path.join(reencoded_training_data_path, 'enc_displacements.dmat')
        energies_path = os.path.join(reencoded_training_data_path, 'energies.dmat')
        
        displacements = my_utils.read_double_dmat_to_numpy(displacements_path)
        encoded_displacements = my_utils.read_double_dmat_to_numpy(enc_displacements_path)
        energies = my_utils.read_double_dmat_to_numpy(energies_path)

    if use_extra_samples:
        sampled_training_data_path = os.path.join(model_root, 'augmented_training_data/sampled/')
        displacements_path = os.path.join(sampled_training_data_path, 'displacements.dmat')
        enc_displacements_path = os.path.join(sampled_training_data_path, 'enc_displacements.dmat')
        energies_path = os.path.join(sampled_training_data_path, 'energies.dmat')
        
        displacements = numpy.append(displacements, my_utils.read_double_dmat_to_numpy(displacements_path), axis=0)
        encoded_displacements = numpy.append(encoded_displacements, my_utils.read_double_dmat_to_numpy(enc_displacements_path), axis=0)
        energies = numpy.append(energies, my_utils.read_double_dmat_to_numpy(energies_path), axis=0)

    return displacements, encoded_displacements, energies

def sparse_learn_weights_l1(model_root, use_reencoded=False, use_extra_samples=False):
    """use_reencoded will use the displacements that have been encoded and then decoded after
    training the model. use_extra_samples uses the gaussian sampled poses around the training data"""
    
    V, T, F = my_utils.read_MESH_to_numpy(os.path.join(model_root, 'tets.mesh'))
    print("Computing volumes...")
    volumes = my_utils.compute_element_volumes(V, T)
    
    # Scale to [0, 1]
    max_vol = numpy.max(volumes)
    volumes /= max_vol
    # numpy.set_printoptions(threshold=numpy.inf)
    # print(volumes)

    displacements, encoded_displacements, energies = load_displacements_and_energy(model_root, use_reencoded, use_extra_samples)

    # Normalize energy
    energies = energies / numpy.apply_along_axis(numpy.linalg.norm, 1, energies)[:, None]

    flatten_data, unflatten_data = my_utils.get_flattners(energies)
    use_binary = False
    if use_binary:
        ae_decode, hist = binary_discrete_nn_analysis(
                                        encoded_displacements, flatten_data(energies),
                                        [], [],
                                        activation='elu',
                                        epochs=70,
                                        batch_size=512,#len(energies),#100,
                                        layers=[200,50], # [200, 200, 50] First two layers being wide seems best so far. maybe an additional narrow third 0.0055 see
                                
                                        do_fine_tuning=False,
                                        model_root=model_root,
                                        autoencoder_config=None,
                                        energy_model_config={'enabled':False},
                                    )
    else:
        ae_decode, hist = discrete_nn_analysis(
                                        encoded_displacements, flatten_data(energies),
                                        [], [],
                                        volumes,
                                        activation='elu',
                                        epochs=400,
                                        batch_size=256,#len(energies),#100,
                                        layers=[50,50,3], # [200, 200, 50] First two layers being wide seems best so far. maybe an additional narrow third 0.0055 see
                                
                                        do_fine_tuning=False,
                                        model_root=model_root,
                                        autoencoder_config=None,
                                        energy_model_config={'enabled':False},
                                    )
        

def learn_direct_energy(model_root, use_reencoded=False, use_extra_samples=False):
    """use_reencoded will use the displacements that have been encoded and then decoded after
    training the model. use_extra_samples uses the gaussian sampled poses around the training data"""
    
    displacements, encoded_displacements, energies = load_displacements_and_energy(model_root, use_reencoded, use_extra_samples)

    energies = numpy.sum(energies, axis=1, keepdims=True)# / 1000

    ae_decode, hist = fit_nn_direct_energy(
                            encoded_displacements, energies,
                            [], [],
                            activation='elu',
                            epochs=200,
                            batch_size=512,#len(energies),#100,
                            layers=[200,500], # [200, 200, 50] First two layers being wide seems best so far. maybe an additional narrow third 0.0055 see
                    
                            do_fine_tuning=False,
                            model_root=model_root,
                            autoencoder_config=None,
                            energy_model_config={'enabled':False},
                        )


def main():
    sparse_learn()
    # build_energy_model('/home/lawson/Workspace/research-experiments/fem-sim/models/x-test-small', 20, 40)

# TODO refactor this..
# Include it in the build model pipeline
def build_energy_model(model_root, energy_model_config):
    """energy basis"""

    # TODO SAVE ENERGY MODEL
    base_path = model_root
    training_data_path = os.path.join(base_path, 'training_data/training')
    validation_data_path = os.path.join(base_path, 'training_data/validation')
    
    # Energies
    energies = my_utils.load_energy_dmats_to_numpy(training_data_path)
    if os.path.exists(validation_data_path):
        energies_test = my_utils.load_energy_dmats_to_numpy(validation_data_path)
    else:
        energies_test = None
    flatten_data, unflatten_data = my_utils.get_flattners(energies)

    energies = flatten_data(energies)#/10000
    # print(energies[100])
    energies_test = flatten_data(energies_test)

    if not os.path.exists(os.path.join(base_path, 'pca_results/')):
        os.makedirs(os.path.join(base_path, 'pca_results/'))
    basis_path = os.path.join(base_path, 'pca_results/energy_pca_components.dmat')
    tet_path = os.path.join(base_path, 'pca_results/energy_indices.dmat')
    start = time.time()
    basis_opt(
        energy_model_config,
        model_root,
        basis_path,
        tet_path,
        energies,
        energies_test,
        energy_model_config['pca_dim'],
        energy_model_config['num_sample_tets'],
        True,
        target_mins=energy_model_config['target_anneal_mins'],
        brute_force_its=energy_model_config['brute_force_iterations']
    )
    print("Energy optimization took", time.time() - start)


def basis_opt(energy_model_config, model_root, basis_output_path, index_output_path, samples, samples_test, pca_dim, num_tets, do_save, target_mins=0.5, brute_force_its=100):#scale=1.0, tmax=60.0, tmin=0.0005, n_steps=10000):
    start = time.time()
    n_tets_sampled = num_tets
    # energy_pca, pca_encode, pca_decode, expl_var, mse = pca_analysis(samples, pca_dim)
    U, explained_variance_ratio, encode, decode = pca_no_centering(samples, pca_dim)

    if do_save:
        my_utils.save_numpy_mat_to_dmat(basis_output_path, numpy.ascontiguousarray(U))

    decoded_energy = decode(encode(samples))
    # print(decoded_energy)
    # decoded_energy = numpy.maximum(decoded_energy, 0, decoded_energy)
    if samples_test is not None:
        decoded_test_energy = decode(encode(samples_test))
    # print(numpy.array(list(zip(numpy.sum(samples_test, axis=1), numpy.sum(decoded_test_energy, axis=1)) ))    )

    # print('train mse: ', mean_squared_error(samples, decoded_energy))    
    pca_train_mse = mean_squared_error(samples, decoded_energy)
    print('train mse: ', pca_train_mse)    
    if samples_test is not None:
        print('test mse:', mean_squared_error(samples_test, decoded_test_energy))    
    print('explained var', sum(explained_variance_ratio))
    
    Es = samples #* 10e4
    # print(Es)
    # print(U)
    Es_summed = numpy.sum(Es, axis=1)

    UTU = U.transpose() @ U
    UTE = U.transpose() @ Es.T
    U_sum = numpy.sum(U, axis=0)
    
    training_log = {'best_energy': []}

    # sol = numpy.linalg.solve(U_bar.T @ U_bar, U_bar.T)
    # x = sol[0]
    # E_stars = (U_sum.T @ x) @ E_bars
    ##### Brute force
    n_random = brute_force_its
    print("\nDoing", n_random, "random iterations...")
    min_mse = 1e100
    best_sample = None
    for i in range(n_random):
        if((i + 1) % 100 == 0):
            print("Iteration:", i + 1)
        tet_sample = numpy.random.choice(U.shape[0], num_tets, replace=False)

        U_bar = U[tet_sample, :]
        E_bars = Es[:, tet_sample]
        # print('rank ', numpy.linalg.matrix_rank(U_bar.T))

        start_solve = time.time()
        if True:
            sol = numpy.linalg.lstsq(U_bar, E_bars.T)
            alphas = sol[0]
            # completed_energies = (U @ alphas).T
            # print('solve took ', time.time() - start_solve)
            # print("solve took", time.time()-start)
            start = time.time()
            # completion_mse = mean_squared_error(completed_energies, Es)
            completion_mse = 1.0/(Es.shape[0] * Es.shape[1]) * (numpy.trace(alphas.transpose() @ UTU @ alphas) - numpy.trace(alphas.transpose() @ UTE)) 
            # print('mse took ', time.time() - start)
        else:
            sol = numpy.linalg.solve(U_bar.T @ U_bar, U_bar.T)
            
            E_stars = (U_sum.T @ sol) @ E_bars.T
            completion_mse = mean_squared_error(E_stars, Es_summed) # Measuring summed energy loss seems to be worse

            # sol = numpy.linalg.lstsq(U_bar, E_bars.T)
            # alphas = sol[0]
            # completed_energies = (U @ alphas).T
            # completion_mse_old = mean_squared_error(numpy.sum(completed_energies, axis=1), Es_summed)

            # print("diff:", completion_mse - completion_mse_old)
        # print('total took ', time.time() - start_solve)
        if completion_mse < min_mse:
            min_mse = completion_mse
            best_sample = tet_sample
            print('min mse (minus constant):', min_mse)

        training_log['best_energy'].append(min_mse)
    a = best_sample

    ## start of anneal (comment out optionally)
    from simanneal import Annealer
    class Problem(Annealer):
        # pass extra data (the distance matrix) into the constructor
        def __init__(self, state, min_mse):
            super(Problem, self).__init__(state)  # important!
            self.min_mse = min_mse

        def move(self):
            # TODO perhaps a better state transition would be to swap in neighboring tets

            #self.state = numpy.random.choice(U.shape[0], n_tets_sampled, replace=False)
            for _ in range(1):
                i = numpy.random.randint(0, n_tets_sampled)
                while True:
                    new_tet = numpy.random.randint(0, len(Es[0]))
                    if new_tet in self.state:
                        pass
                        # print('yikes')
                    else:
                        break

                self.state[i] = new_tet

        def energy(self):
            """Calculates the length of the route."""
            U_bar = U[self.state, :]
            E_bars = Es[:, self.state]

            # start = time.time()
            if True:
                sol = numpy.linalg.lstsq(U_bar, E_bars.T)
                alphas = sol[0]
                # completed_energies = (U @ sol[0]).T
                # print("solve took", time.time()-start)
                # check_samples = numpy.random.choice(Es.shape[0], 100, replace=False)
                # Full
                # completion_mse = mean_squared_error(completed_energies, Es)*scale
                completion_mse = 1.0/(Es.shape[0] * Es.shape[1]) * (numpy.trace(alphas.transpose() @ UTU @ alphas) - numpy.trace(alphas.transpose() @ UTE)) 
            else:
                sol = numpy.linalg.solve(U_bar.T @ U_bar, U_bar.T)
                
                E_stars = (U_sum.T @ sol) @ E_bars.T
                completion_mse = mean_squared_error(E_stars, Es_summed)
            
            if completion_mse < self.min_mse:
                self.min_mse = completion_mse
            training_log['best_energy'].append(self.min_mse) 

            return completion_mse #* 1000000.0

    init_state = a #numpy.random.choice(U.shape[0], n_tets_sampled, replace=False)
    prob = Problem(init_state, min_mse)
    
    prob.Tmax = 1 #tmax  # Max (starting) temperature
    prob.Tmin = 0.00001#tmin      # Min (ending) temperature
    prob.steps = 10000 #n_steps   # Number of iterations
    prob.updates = prob.steps / 100 
    print("\nDetermining annealing schedule...")
    # auto_schedule = prob.auto(minutes=target_mins, steps=20) 
    # prob.set_schedule(auto_schedule)

    print("\nRunning simulated annealing...")
    a, b = prob.anneal()
    print("Done.")
    print("\nSelected tets:", a)

    print('\nmin mse (minus constant):', b)

    U_bar = U[a, :]
    E_bars = Es[:, a]
    sol = numpy.linalg.lstsq(U_bar, E_bars.T)
    completed_energies = (U @ sol[0]).T
    completion_mse = mean_squared_error(completed_energies, Es)
    print('Final reduced energy mse:', completion_mse, 'vs pca minimum of:', pca_train_mse)
    print()
    ######## End of anneal

    training_log["final_tets"] = a.tolist()
    training_log["final_mse"] = completion_mse
    training_log["pca_mse"] = pca_train_mse
    training_log["config"] = energy_model_config
    training_log["total_time"] = time.time() - start

    if do_save:
        my_utils.save_numpy_mat_to_dmat(index_output_path, numpy.ascontiguousarray(numpy.array(a,dtype=numpy.int32)))
        with open(os.path.join(model_root, 'energy_training_history.json'), 'w') as f:
            json.dump(training_log, f, indent=2)



def generate_model(
    model_root, # This is the root of the standard formatted directory created by build-model.py
    learning_config, # Comes from the training config file
    ):
    autoencoder_config = learning_config['autoencoder_config']
    energy_model_config = learning_config['energy_model_config']
    save_objs = learning_config['save_objs']
    train_in_full_space = False
    record_full_mse_each_epoch = False

    training_data_path = os.path.join(model_root, 'training_data/training')
    validation_data_path = os.path.join(model_root, 'training_data/validation')
    # Loading the rest pose
    base_verts, face_indices = my_utils.load_base_vert_and_face_dmat_to_numpy(training_data_path)
    base_verts_eig = p2e(base_verts)
    face_indices_eig = p2e(face_indices)

    # Loading displacements for training data
    displacements = my_utils.load_displacement_dmats_to_numpy(training_data_path)
    if os.path.exists(validation_data_path):
        test_displacements = my_utils.load_displacement_dmats_to_numpy(validation_data_path)
    else:
        test_displacements = None
    flatten_data, unflatten_data = my_utils.get_flattners(displacements)
    displacements = flatten_data(displacements)
    test_displacements = flatten_data(test_displacements)

    # Do the training
    pca_ae_train_dim = autoencoder_config['pca_layer_dim']
    pca_compare_dims = autoencoder_config['pca_compare_dims']
    ae_dim = autoencoder_config['ae_encoded_dim']
    ae_epochs = autoencoder_config['training_epochs']
    batch_size = autoencoder_config['batch_size'] if autoencoder_config['batch_size'] > 0 else len(displacements)
    layers = autoencoder_config['non_pca_layer_sizes']
    do_fine_tuning = autoencoder_config['do_fine_tuning']
    activation = autoencoder_config['activation']
    save_pca_components = True
    save_autoencoder = True

    training_results = { 
        'autoencoder': {},
        'pca': {},
    }

    # Normal low dim pca first
    for pca_dim in pca_compare_dims:
        U, explained_var, pca_encode, pca_decode = pca_no_centering(displacements, pca_dim)

        encoded_pca_displacements = pca_encode(displacements)
        decoded_pca_displacements = pca_decode(encoded_pca_displacements)
        if test_displacements is not None:
            encoded_pca_test_displacements = pca_encode(test_displacements)
            decoded_pca_test_displacements = pca_decode(encoded_pca_test_displacements)

        if save_pca_components:
            pca_results_filename = os.path.join(model_root, 'pca_results/pca_components_' + str(pca_dim) + '.dmat')
            print('Saving pca results to', pca_results_filename)
            my_utils.create_dir_if_not_exist(os.path.dirname(pca_results_filename))
            my_utils.save_numpy_mat_to_dmat(pca_results_filename, numpy.ascontiguousarray(U))

            training_mse = mean_squared_error(flatten_data(decoded_pca_displacements), flatten_data(displacements))
            if test_displacements is not None:
                validation_mse = mean_squared_error(flatten_data(decoded_pca_test_displacements), flatten_data(test_displacements))
            training_results['pca'][str(pca_dim) + '-components'] = {}
            training_results['pca'][str(pca_dim) + '-components']['training-mse'] = training_mse
            training_results['pca'][str(pca_dim) + '-components']['explained_var'] = numpy.sum(explained_var)
            if test_displacements is not None:
                training_results['pca'][str(pca_dim) + '-components']['validation-mse'] = validation_mse

            print(str(pca_dim) + ' training MSE: ', training_mse)
            print(str(pca_dim) + ' explained variance: ', numpy.sum(explained_var))
            print()


    # High dim pca to train autoencoder
    pca_start_time = time.time()
    U_ae, explained_var, high_dim_pca_encode, high_dim_pca_decode = pca_no_centering(displacements, pca_ae_train_dim)
    pca_train_time = time.time() - pca_start_time

    if train_in_full_space:
        high_dim_pca_encode = lambda x: x
        high_dim_pca_decode = lambda x: x
    encoded_high_dim_pca_displacements = high_dim_pca_encode(displacements)
    if test_displacements is not None:
        encoded_high_dim_pca_test_displacements = high_dim_pca_encode(test_displacements)
    else:
        encoded_high_dim_pca_test_displacements = None

    from keras.callbacks import Callback 
    class MyHistoryCallback(Callback):
        """Callback that records events into a `History` object.
        This callback is automatically applied to
        every Keras model. The `History` object
        gets returned by the `fit` method of models.
        """

        def on_train_begin(self, logs=None):
            self.epoch = []
            self.history = {}

        def on_epoch_end(self, epoch, logs=None):
            if record_full_mse_each_epoch:
                mse = mean_squared_error(high_dim_pca_decode(self.model.predict(encoded_high_dim_pca_displacements)), displacements)
                val_mse = mean_squared_error(high_dim_pca_decode(self.model.predict(encoded_high_dim_pca_test_displacements)), test_displacements)
                self.history.setdefault('mse', []).append(mse)
                self.history.setdefault('val_mse', []).append(val_mse)
                print()
                print("Mean squared error: ", mse)
                print("Val Mean squared error: ", val_mse)

            logs = logs or {}
            self.epoch.append(epoch)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

    my_hist_callback = MyHistoryCallback()

    ae_pca_basis_path = os.path.join(model_root, 'pca_results/ae_pca_components.dmat')
    print('Saving pca results to', ae_pca_basis_path)
    my_utils.save_numpy_mat_to_dmat(ae_pca_basis_path, numpy.ascontiguousarray(U_ae))
    ae_encode, ae_decode, ae_train_time = autoencoder_analysis(
                                    # displacements, # Uncomment to train in full space
                                    # test_displacements, 
                                    encoded_high_dim_pca_displacements,
                                    encoded_high_dim_pca_test_displacements,
                                    activation=activation,
                                    latent_dim=ae_dim,
                                    epochs=ae_epochs,
                                    batch_size=batch_size,
                                    layers=layers, # [200, 200, 50] First two layers being wide seems best so far. maybe an additional narrow third 0.0055 see
                                    # pca_basis=U_ae,
                                    do_fine_tuning=do_fine_tuning,
                                    model_root=model_root,
                                    autoencoder_config=autoencoder_config,
                                    callback=my_hist_callback,
                                )

    # decoded_autoencoder_displacements = ae_decode(ae_encode(displacements))
    # decoded_autoencoder_test_displacements = ae_decode(ae_encode(test_displacements))
    decoded_autoencoder_displacements = high_dim_pca_decode(ae_decode(ae_encode(high_dim_pca_encode(displacements))))
    if test_displacements is not None:
        decoded_autoencoder_test_displacements = high_dim_pca_decode(ae_decode(ae_encode(high_dim_pca_encode(test_displacements))))
    ae_training_mse = mean_squared_error(flatten_data(decoded_autoencoder_displacements), flatten_data(displacements))
    training_results['autoencoder']['training-mse'] = ae_training_mse
    if test_displacements is not None:
        training_results['autoencoder']['validation-mse'] = mean_squared_error(flatten_data(decoded_autoencoder_test_displacements), flatten_data(test_displacements))

    print('Finale ae training MSE:', ae_training_mse)
    training_results['autoencoder']['ae_training_time_s'] = ae_train_time
    training_results['autoencoder']['pca_training_time_s'] = pca_train_time
    # TODO output energy loss as well
    with open(os.path.join(model_root, 'training_results.json'), 'w') as f:
        json.dump(training_results, f, indent=2)

    history_path = os.path.join(model_root, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(my_hist_callback.history, f, indent=2)

    if save_objs:
        print("Saving objs of decoded training data...")
        obj_dir = os.path.join(model_root, 'decoded_training_objs/')
        if not os.path.exists(obj_dir):
            os.makedirs(obj_dir)

        for i, dec_displ in enumerate(decoded_autoencoder_displacements):
            decoded_verts = base_verts + dec_displ.reshape((len(base_verts), 3))
            decoded_verts_eig = p2e(decoded_verts)

            obj_path = os.path.join(obj_dir, "decoded_" + str(i) + ".obj")
            igl.writeOBJ(obj_path, decoded_verts_eig, face_indices_eig)
    

if __name__ == "__main__":
    main()
