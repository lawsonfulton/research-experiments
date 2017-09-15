#import numpy
import autograd
import autograd.numpy as numpy

from scipy import optimize

from viz import render

def construct_P_matrices(springs, n_points, d):
    return numpy.array([
        numpy.concatenate((
            numpy.concatenate((numpy.zeros((d, d * s_k[0])), numpy.identity(d), numpy.zeros((d, d * n_points - d * (s_k[0] + 1)))), axis=1),
            numpy.concatenate((numpy.zeros((d, d * s_k[1])), numpy.identity(d), numpy.zeros((d, d * n_points - d * (s_k[1] + 1)))), axis=1),
        )) for s_k in springs])

def main():
    ### Setup

    # Constants
    d = 2  # dimensions
    I = numpy.identity(d)
    B = numpy.concatenate((I, -I), axis=1)  # Difference matrix

    # Simulation Parameters
    spring_const = 10.0 # Technically could vary per spring
    h = 0.01
    mass = 5
    # Initial conditions
    starting_stretch = 1#0.6

    # starting_points = numpy.array([
    #     [0.4, 0.5],
    #     [0.6, 0.5],
    #     [0.8, 0.5],
    #     [1, 0.5]
    # ])
    starting_points = numpy.array([
        [0,1],
        [0,0],
        [1,1],
        [1,0],
        [2,1],
        [2,0],
        # [0,2],
        # [0,-1]
    ]) * 0.1 + 0.5
    n_points = len(starting_points) # Num points
    q_initial = starting_points.flatten()

    pinned_points = [0, 1]#, 6, 7]

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
        # (6, 2),
        # (7, 3)
    ]

    n_springs = len(springs)
    P_matrices = construct_P_matrices(springs, n_points, d)
    
    all_spring_offsets = (B @ (P_matrices @ q_initial).T).T
    rest_lens = numpy.linalg.norm(all_spring_offsets, axis=1) * starting_stretch

    mass_matrix = numpy.identity(len(q_initial)) * mass # Mass matrix
    external_forces = numpy.array([0, -1] * n_points)
    
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
        return D1_Ld(cur_q, new_q) + D2_Ld(prev_q, cur_q) + mass_matrix @ external_forces

    jac_DEL = autograd.jacobian(DEL, 0)

    ### Simulation
    prev_q = q_initial
    cur_q = q_initial
    while True:
        sol = optimize.root(DEL, cur_q, method='broyden1', args=(cur_q, prev_q))#, jac=jac_DEL) # Note numerical jacobian seems much faster
        prev_q = cur_q
        cur_q = sol.x

        # Fix the position of some points. Is this valid?? not really...
        for p in pinned_points:
            index = p * d
            cur_q[index] = q_initial[index]
            cur_q[index + 1] = q_initial[index + 1]

        render(cur_q, springs, save_frames=False)

if __name__ == '__main__':
    main()
