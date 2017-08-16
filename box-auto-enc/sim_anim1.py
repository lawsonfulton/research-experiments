import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.misc import derivative as numeric_derivative
import matplotlib.animation as animation

import boxes

use_autoencoder = True

if use_autoencoder:
    import numpy as np
    from keras_learn import load_autoencoder, jacobian_output_wrt_input
    import keras.backend as K
    import tensorflow as tf
else:
    from autograd import grad, jacobian
    import autograd.numpy as np # Need to override numpy with autograd numpy if using explicit mapping
    boxes.np = np


class BoxSim:
    def __init__(self, use_autoencoder=False):
        self.use_autoencoder = use_autoencoder

        self.time_elapsed = 0.0
        self.world_force = np.array([2.0, 15.8, 0.0, -9.8, 0.0, -9.8, 0.0, -9.8])
        starting_pos = [0.0, 0.0]
        starting_theta = 0.0

        if use_autoencoder:
            model_path = "models/contractive loss 250 epochs.h5"
            self.autoencoder, self.encoder, self.decoder = load_autoencoder(model_path)
            #self.jac_x_wrt_q = jacobian_output_wrt_input(self.decoder)

            starting_box = boxes.generate_samples(offsets=[starting_pos], thetas=[starting_theta])
            gen_pos = self.encoder.predict(starting_box)[0] # need to get the encoded starting position
        else:
            self.jac_x_wrt_q = jacobian(boxes.explicit_decode)
            gen_pos = np.array([*starting_pos, starting_theta])
        
        gen_vel = np.array([0.0, 0.0, 0.0])
        self.state = np.array([*gen_pos, *gen_vel])

    def decode_q_to_x(self, q):
        if self.use_autoencoder:
            return self.decoder.predict(np.array([q]))[0]
        else:
            return boxes.explicit_decode(q)

    def numeric_jacobian(self, q):
        jac = np.zeros((8,3))
        for x_i in range(8):
            for q_i in range(3):
                def f_xi_qi(a):
                    new_q = q.copy()
                    new_q[q_i] = a
                    return self.decode_q_to_x(new_q)[x_i]

                jac[x_i][q_i] = numeric_derivative(f_xi_qi, q[q_i], dx=1e-6)

        return jac

    def test_jacobian(self):
        starting_pos = [0.0, 0.0]
        starting_theta = 0.0
        if self.use_autoencoder:
            q = self.encoder.predict(boxes.generate_samples(offsets=[starting_pos], thetas=[starting_theta]))[0]
        else:
            q = np.array([*starting_pos, starting_theta])

        print("Actual x: ", self.decode_q_to_x(q))

        cum_error = 0.0
        for x_i in range(8):
            for q_i in range(3):
                # x_i = 0
                # q_i = 0

                def f_xi_qi(a):
                    new_q = q.copy()
                    new_q[q_i] = a
                    return self.decode_q_to_x(new_q)[x_i]

                numeric_d = numeric_derivative(f_xi_qi, q[q_i], dx=1e-6)
                actual_d = self.jac_x_wrt_q(q)[x_i][q_i]
                error = np.abs(numeric_d - actual_d)
                cum_error += error

                print("partial of x", x_i, " wrt q", q_i, sep = "")
                print("Numeric derivative:", numeric_d)
                print("Acutal derivative:", actual_d)
                print("Difference:", error)

                print()
        print("Cumulative error: ", cum_error)

    def positions(self):
        if self.use_autoencoder:
            return boxes.box_from_vec(self.decoder.predict(np.array([self.state[0:3]]))[0])
        else:
            return boxes.box_from_vec(boxes.explicit_decode(self.state[0:3]))

    # def energy(self):
    #     """compute the energy of the current state"""


    def dstate_dt(self, state, t):
        """compute the derivative of the given state"""
        dsdt = np.zeros_like(state)

        q = state[0:3]
        dqdt = state[3:6]


        #dvdt = mass_matrix_inv * self.jac_x_wrt_q(q) * mass_matrix * self.world_force
        jacxq = self.jac_x_wrt_q(q)
        #jacxq = self.numeric_jacobian(q)
        dvdt = np.dot(self.world_force, jacxq)
        
        dsdt[0:3] = dqdt # dpos/dt = vel
        dsdt[3:6] = dvdt

        return dsdt

    def my_integrate(self, state, dt):
        q = state[0:3] # generalized coordinates
        qv = state[3:6] # generalized velocity

        #jacxq = self.jac_x_wrt_q(q) # jacobian of world coords wrt generalized coords at q
        jacxq = self.numeric_jacobian(q)
        qa = np.dot(self.world_force, jacxq) # generalized acceleration == generalized forces (mass = 1 everywhere)
        
        new_qv = qv + dt * qa
        new_q = q + dt * qv

        new_state = np.array([*new_q, *new_qv])
        return new_state



    def step(self, dt):
        """execute one time step of length dt and update state"""
        #self.state = integrate.odeint(self.dstate_dt, self.state, [0, dt])[1]
        self.state = self.my_integrate(self.state, dt)
        self.time_elapsed += dt

def simulate():
    boxsim = BoxSim(use_autoencoder=use_autoencoder)
    # boxsim.test_jacobian()
    # exit()
    dt = 1.0 / 2500

    #------------------------------------------------------------
    # set up figure and animation
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-10, 10), ylim=(-10, 10))
    ax.grid()

    #line, = ax.plot([], [], 'o-', lw=2)
    patch = boxes.draw_box([[0,0],[0,0]], ax)
    points, = ax.plot([], [], 'ro', ms=3)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

    def init():
        """initialize animation"""
        #line.set_data([], [])

        time_text.set_text('')
        energy_text.set_text('')
        return patch, points, time_text, energy_text

    def animate(i):
        """perform animation step"""
        boxsim.step(dt)

        positions = boxsim.positions()
        #line.set_data(positions[:,0], positions[:,1])
        patch.set_xy(positions)
        points.set_data(positions[:,0], positions[:,1])
        time_text.set_text('time = %.1f' % boxsim.time_elapsed)
        # energy_text.set_text('energy = %.3f J' % boxsim.energy())
        return patch, points, time_text, energy_text


    interval = 1000/20
    print("Saving")
    ani = animation.FuncAnimation(fig, animate, frames=150,
                                  interval=interval, blit=True, init_func=init)#.save('simulation.gif', writer='imagemagick')


    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    #ani.save('double_pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()
    

    return ani

if __name__ == "__main__":
    simulate()