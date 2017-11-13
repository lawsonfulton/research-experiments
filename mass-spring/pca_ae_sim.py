#import numpy
import autograd
import autograd.numpy as numpy

from scipy import optimize

import pickle

from viz import render


## CPU is probably faster for forward pass
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# from learning import load_autoencoder


def construct_P_matrices(springs, n_points, d):
    return numpy.array([
        numpy.concatenate((
            numpy.concatenate((numpy.zeros((d, d * s_k[0])), numpy.identity(d), numpy.zeros((d, d * n_points - d * (s_k[0] + 1)))), axis=1),
            numpy.concatenate((numpy.zeros((d, d * s_k[1])), numpy.identity(d), numpy.zeros((d, d * n_points - d * (s_k[1] + 1)))), axis=1),
        )) for s_k in springs])

def main():
    ### Setup
    # autoencoder, encoder, decoder = load_autoencoder('models/12 17PM November 08 2017.h5')#models/2.7e-06.h5')
    # def encode(q):
    #     return encoder.predict(numpy.array([q]))[0].astype(numpy.float64)

    # def decode(z):
    #     return decoder.predict(numpy.array([z]))[0].astype(numpy.float64)
    encode, decode = train_model()

    # Constants
    d = 2  # dimensions
    I = numpy.identity(d)
    B = numpy.concatenate((I, -I), axis=1)  # Difference matrix

    # Simulation Parameters
    spring_const = 10.0 # Technically could vary per spring
    h = 0.005
    mass = 0.05
    # Initial conditions
    starting_stretch = 1#0.6

    starting_points = numpy.array([
        [0,1],
        [0,0],
        [1,1],
        [1,0],
        [2,1],
        [2,0],
        [3,1],
        [3,0],
        [4,1],
        [4,0],
    ]) * 0.1 + 0.3
    n_points = len(starting_points) # Num points
    q_initial = starting_points.flatten()
    z_initial = encode(q_initial)
    print("HIOIIIIII")
    print(q_initial)
    print(z_initial)
    print(decode(z_initial))

    pinned_points = numpy.array([0, 1])
    q_mask = numpy.ones(n_points * d, dtype=bool)
    q_mask[numpy.concatenate([pinned_points * d + i for i in range(d)])] = False

    springs = [
        (0, 2),
        (0, 3),
        (2, 3),
        (1, 2),
        (1, 3),
        (2, 4),
        (2, 3),
        (2, 5),
        (3, 5),
        (3, 4),
        (4, 5),
        (4, 6),
        (4, 7),
        (5, 6),
        (5, 7),
        (6, 7),

        (6, 8),
        (6, 9),
        (7, 8),
        (7, 9),
        (8, 9),
    ]

    n_springs = len(springs)
    P_matrices = construct_P_matrices(springs, n_points, d)
    
    all_spring_offsets = (B @ (P_matrices @ q_initial).T).T
    rest_lens = numpy.linalg.norm(all_spring_offsets, axis=1) * starting_stretch

    mass_matrix = numpy.identity(len(q_initial)) * mass # Mass matrix
    external_forces = numpy.array([0, -9.8] * n_points)
    
    def kinetic_energy(q_k, q_k1):
        """ Profile this to see if using numpy.dot is different from numpy.matmul (@)"""

        d_q = q_k1 - q_k
        energy = 1.0 / (2 * h ** 2) * d_q.T @ mass_matrix @ d_q

        return energy

    def potential_energy(q_k, q_k1):
        q_tilde = 0.5 * (q_k + q_k1)

        # sum = 0.0
        # for i in range(len(rest_lens)): # TODO I might be able to do this simply with @ (see all_spring_offsets)
        #     P_i = P_matrices[i]
        #     l_i = rest_lens[i]

        #     d_q_tilde_i_sq = q_tilde.T @ P_i.T @ B.T @ B @ P_i @ q_tilde
        #     sum += (1.0 - 1.0 / l_i * numpy.sqrt(d_q_tilde_i_sq)) ** 2

        # Optimized but ugly version
        sum = numpy.sum(
            (1.0 - (1.0 / rest_lens) * numpy.sqrt(numpy.einsum('ij,ij->i', q_tilde.T @ P_matrices.transpose((0,2,1)) @ B.T, (B @ P_matrices @ q_tilde)))) ** 2
        )

        return 0.5 * spring_const * sum

    def discrete_lagrangian(q_k, q_k1):
        return kinetic_energy(q_k, q_k1) - potential_energy(q_k, q_k1)

    D1_Ld = autograd.grad(discrete_lagrangian, 0)  # (q_t, q_t+1) -> R^N*d
    D2_Ld = autograd.grad(discrete_lagrangian, 1)  # (q_t-1, q_t) -> R^N*d


    # Want D1_Ld + D2_Ld = 0
    # Do root finding
    def DEL(new_q, cur_q, prev_q):
        # SUPER hacky way of adding constrained points
        for i in pinned_points:
            new_q = numpy.insert(new_q, i*d, q_initial[i*d])
            new_q = numpy.insert(new_q, i*d+1, q_initial[i*d+1])

        res = D1_Ld(cur_q, new_q) + D2_Ld(prev_q, cur_q) + mass_matrix @ external_forces

        # SUPER hacky way of adding constrained points
        return res[q_mask]

    jac_DEL = autograd.jacobian(DEL, 0)

    def latent_DEL(new_z, cur_q, prev_q):
        new_q = decode(new_z)
        res = D1_Ld(cur_q, new_q) + D2_Ld(prev_q, cur_q) + mass_matrix @ external_forces

        return res

    def latent_DEL_objective(new_z, cur_q, prev_q):
        res = latent_DEL(new_z, cur_q, prev_q)

        return res.T @ res

    ### Simulation
    q_history = []
    save_freq = 1000
    current_frame = 0
    output_path = 'configurations'

    prev_q = q_initial
    cur_q = q_initial
    prev_z = z_initial
    cur_z = z_initial
    while True:

        #sol = optimize.root(latent_DEL, cur_z, method='broyden1', args=(cur_q, prev_q))#, jac=jac_DEL) # Note numerical jacobian seems much faster
        sol = optimize.minimize(latent_DEL_objective, cur_z, args=(cur_q, prev_q), method='L-BFGS-B', options={'gtol': 1e-6, 'eps': 1e-06, 'disp': False})
        prev_z = cur_z
        cur_z = sol.x

        prev_q = cur_q
        cur_q = decode(cur_z)
        render(cur_q * 10, springs, save_frames=True)

        # if save_freq > 0:
        #     current_frame += 1
        #     q_history.append(cur_q)

        #     if current_frame % save_freq == 0:
        #         with open(output_path, 'wb') as f:
        #             pickle.dump(q_history, f)


def decompose_ae(autoencoder):
    import keras
    from keras.layers import Input, Dense
    from keras.models import Model, load_model
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

def train_model():
    import time
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
    encoded_dim = 2

    # Single autoencoder
    # initializer = keras.initializers.RandomUniform(minval=0.0, maxval=0.01, seed=5)
    # bias_initializer = initializer
    import keras
    from keras.layers import Input, Dense
    from keras.models import Model, load_model

    activation = keras.layers.advanced_activations.LeakyReLU(alpha=0.3) #'relu'
    
    input = Input(shape=(len(train_data[0]),))
    output = input
    
    output = Dense(32, activation=activation)(output)
    output = Dense(16, activation=activation)(output)
    output = Dense(encoded_dim, activation=activation, name="encoded")(output)
    output = Dense(16, activation=activation)(output)
    output = Dense(32, activation=activation)(output)
    
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
        epochs=800,#800,
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


    print("Total model time: ", time.time() - model_start_time)

    autoencoder,encoder, decoder = decompose_ae(autoencoder)
    # Display
    def decode(z):
        ae_decoded = denormalize(decoder.predict(numpy.array([z])))
        return (numpy_base_verts + pca.inverse_transform(ae_decoded))[0]#.reshape(test_size, num_verts, 2)[0]

    def encode(q):
        ae_encoded = encoder.predict(normalize(pca.transform(numpy.array([q - numpy_base_verts]))))
        return ae_encoded[0]

    return encode, decode

if __name__ == '__main__':
    main()
