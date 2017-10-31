from sklearn.decomposition import PCA
import autograd
import autograd.numpy as numpy

from scipy import optimize

import pickle

from viz import render

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data

# def main():
#     sample_size = 10
#     test_size = 100
#     data = load_data('mass-spring-bar-configurations_small.pickle')
#     test_data = data[:test_size]

#     numpy.random.shuffle(data)
#     train_data = data[:sample_size]

#     pca = PCA(n_components=1)
#     pca.fit(train_data)
    
#     encoded = pca.transform(test_data)
#     decoded = pca.inverse_transform(encoded)

#     c = 0
#     while True:
#         i = c % test_size
#         c += 1
#         viz.render(decoded[i], [])

def construct_P_matrices(springs, n_points, d):
    return numpy.array([
        numpy.concatenate((
            numpy.concatenate((numpy.zeros((d, d * s_k[0])), numpy.identity(d), numpy.zeros((d, d * n_points - d * (s_k[0] + 1)))), axis=1),
            numpy.concatenate((numpy.zeros((d, d * s_k[1])), numpy.identity(d), numpy.zeros((d, d * n_points - d * (s_k[1] + 1)))), axis=1),
        )) for s_k in springs])

def main():
    ### Setup
    sample_size = 100
    data = load_data('mass-spring-bar-configurations_small.pickle')
    numpy.random.shuffle(data)
    train_data = data[:sample_size]

    pca = PCA(n_components=1)
    pca.fit(train_data)

    def encode(q):
        return pca.transform(numpy.array([q]))[0]

    def decode(z):
        return pca.inverse_transform(numpy.array([z]))[0]

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

        # sol = optimize.root(latent_DEL, cur_z, method='broyden1', args=(cur_q, prev_q))#, jac=jac_DEL) # Note numerical jacobian seems much faster
        sol = optimize.minimize(latent_DEL_objective, cur_z, args=(cur_q, prev_q), method='L-BFGS-B')
        prev_z = cur_z
        cur_z = sol.x

        prev_q = cur_q
        cur_q = decode(cur_z)
        render(cur_q, springs, save_frames=True)

        # if save_freq > 0:
        #     current_frame += 1
        #     q_history.append(cur_q)

        #     if current_frame % save_freq == 0:
        #         with open(output_path, 'wb') as f:
        #             pickle.dump(q_history, f)

if __name__ == '__main__':
    main()
