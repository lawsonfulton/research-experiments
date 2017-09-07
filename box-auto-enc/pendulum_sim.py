import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.misc import derivative as numeric_derivative
import scipy
import matplotlib.animation as animation

import pendulums

use_autoencoder = False

if use_autoencoder:
    import numpy as np
    from pendulum_learn import load_autoencoder, jacobian_output_wrt_input
    import keras.backend as K
    import tensorflow as tf
else:
    from autograd import grad, jacobian
    import autograd.numpy as np # Need to override numpy with autograd numpy if using explicit mapping
    pendulums.np = np

# Could make this general but no point right now
class PendulumSim:
    def __init__(self, use_autoencoder=False, model_path=None):
        self.use_autoencoder = use_autoencoder

        self.time_elapsed = 0.0

        self.dqdt2 = 0.0 # for reporting

        if use_autoencoder:
            self.autoencoder, self.encoder, self.decoder = load_autoencoder(model_path)
            self.enc_dim = self.encoder.output_shape[-1] # Dimension of encoded configuration space
            self.world_dim = self.decoder.output_shape[-1] # Dimension of system in world coords

            #self.jac_x_wrt_q = jacobian_output_wrt_input(self.decoder)

            starting_pendulum = pendulums.generate_samples(thetas=[[0.0, 0.0]])
            gen_pos = self.encoder.predict(starting_pendulum)[0] # need to get the encoded starting position
        else:
            self.jac_x_wrt_q = jacobian(pendulums.explicit_decode)
            gen_pos = np.array([0.0, 0.0])
            self.enc_dim = len(gen_pos)
            self.world_dim = len(pendulums.explicit_decode(gen_pos))
        
        self.world_force = np.array([0.0, -9.8] * (self.world_dim // 2))
        gen_vel = np.array([0.0] * self.enc_dim)
        self.state = np.array([*gen_pos, *gen_vel])
        self.prev_state = self.state

    def decode_q_to_x(self, q):
        if self.use_autoencoder:
            return self.decoder.predict(np.array([q]))[0]
        else:
            return pendulums.explicit_decode(q)

    def encode_x_to_q(self, x):
        if self.use_autoencoder:
            return self.encoder.predict(np.array([x]))[0]
        else:
            raise "Not implemented for explicit mapping"

    def numeric_jacobian(self, q):
        jac = np.zeros((self.world_dim, self.enc_dim))
        for x_i in range(self.world_dim):
            for q_i in range(self.enc_dim):
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
            q = self.encoder.predict(pendulums.generate_samples(offsets=[starting_pos], thetas=[starting_theta]))[0]
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

    def generalized_position(self, state):
        return state[0:self.enc_dim]

    def generalized_velocity(self, state):
        return state[self.enc_dim:]

    def positions(self):
        if self.use_autoencoder:
            return pendulums.pendulum_from_vec(self.decoder.predict(np.array([self.generalized_position(self.state)]))[0])
        else:
            return pendulums.pendulum_from_vec(pendulums.explicit_decode(self.generalized_position(self.state)))

    def energy(self, dt):
        """Estimate kinetic and potential energy using world coordinates from current and previous state"""
        x = self.decode_q_to_x(self.generalized_position(self.state))
        prev_x = self.decode_q_to_x(self.generalized_position(self.prev_state))

        vel = (x - prev_x) / dt

        # Assuming m = 1 everywhere
        kinetic = 0.5 * np.dot(vel, vel)
        potential = np.dot(-self.world_force, x)

        return kinetic, potential


    def my_integrate(self, state, dt):
        q = self.generalized_position(state) # generalized coordinates
        qv = self.generalized_velocity(state) # generalized velocity

        D_q = self.decode_q_to_x(q)

        use_exact_jacobian = True
        if use_exact_jacobian:
            jacxq =self.jac_x_wrt_q(q)
        else:
            jacxq = self.numeric_jacobian(q)

        jacxq_dot_qv = np.dot(jacxq, qv)

        v_star = D_q + dt * jacxq_dot_qv

        ###
        # I
        A = np.dot(jacxq.transpose() , jacxq)
        b = np.dot(jacxq.transpose(), np.dot(jacxq, qv) + dt * self.world_force)
        new_qv = scipy.linalg.solve(A, b)

        # II
        use_optimize = True
        if use_optimize:
            def objective(new_q):
                D_new_q = self.decode_q_to_x(new_q)
                diff = D_new_q - v_star
                return 0.5 * np.dot(diff, diff)

            q_0 = q
            res = scipy.optimize.minimize(objective, q_0, method='L-BFGS-B', options={'gtol': 1e-6, 'eps': 1e-06, 'disp': False})
            #res = scipy.optimize.minimize(objective, q_0, jac=jacobian(objective), method='L-BFGS-B', options={'gtol': 1e-8, 'eps': 1e-08, 'disp': False})
            new_q = res.x
        else:
            new_D_q = D_q + dt *  np.dot(jacxq, qv)
            new_q = self.encode_x_to_q(new_D_q)

        new_state = np.array([*new_q, *new_qv])
        #print(new_state)
        return new_state

    def my_integrate_reproj(self, state, dt):
        q = self.generalized_position(state) # generalized coordinates
        qv = self.generalized_velocity(state) # generalized velocity

        D_q = self.decode_q_to_x(q)

        use_exact_jacobian = True
        if use_exact_jacobian:
            jacxq =self.jac_x_wrt_q(q)
        else:
            jacxq = self.numeric_jacobian(q)

        jacxq_dot_qv = np.dot(jacxq, qv)

        v_star = D_q + dt * jacxq_dot_qv

        ###
        # I
        A = np.dot(jacxq.transpose() , jacxq)
        b = np.dot(jacxq.transpose(), np.dot(jacxq, qv) + dt * self.world_force)
        new_qv = scipy.linalg.solve(A, b)

        # II
        use_optimize = True
        if use_optimize:
            def objective(new_q):
                D_new_q = self.decode_q_to_x(new_q)
                diff = D_new_q - v_star
                return 0.5 * np.dot(diff, diff)

            q_0 = q
            res = scipy.optimize.minimize(objective, q_0, method='L-BFGS-B', options={'gtol': 1e-6, 'eps': 1e-06, 'disp': False})
            #res = scipy.optimize.minimize(objective, q_0, jac=jacobian(objective), method='L-BFGS-B', options={'gtol': 1e-8, 'eps': 1e-08, 'disp': False})
            new_q = res.x
        else:
            new_D_q = D_q + dt *  np.dot(jacxq, qv)
            new_q = self.encode_x_to_q(new_D_q)

        new_state = np.array([*new_q, *new_qv])
        #print(new_state)
        return new_state

    def step(self, dt):
        """execute one time step of length dt and update state"""
        #self.state = integrate.odeint(self.dstate_dt, self.state, [0, dt])[1]
        self.prev_state = self.state

        use_reprojection = True
        if use_reprojection:
            self.state = self.my_integrate_reproj(self.state, dt)
        else:
            self.state = self.my_integrate(self.state, dt)

        self.time_elapsed += dt

pendulum_qs = np.array([])

def simulate(model_path=None):
    pendulumsim = PendulumSim(use_autoencoder=use_autoencoder, model_path=model_path)
    # pendulumsim.test_jacobian()
    # exit()
    dt = 1.0 / 800

    #------------------------------------------------------------
    # set up figure and animation
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-10, 10), ylim=(-10, 10))
    ax.grid()

    # ax3d = fig.add_subplot(1,2,2, projection='3d')
    max_d = 0.2
    # ax3d.set_xlim([0,max_d])
    # ax3d.set_ylim([0,max_d])
    # ax3d.set_zlim([0,max_d])
    #line, = ax.plot([], [], 'o-', lw=2)
    patch, points = pendulums.draw_pendulum(np.array([[0,0],[0,0]]), ax)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    # global pendulum_qs
    # pendulum_qs = np.array([pendulumsim.generalized_position()])
    # line, = ax3d.plot(pendulum_qs[0:1,0], pendulum_qs[0:1,1], pendulum_qs[0:1,2])
    def init():
        """initialize animation"""
        #line.set_data([], [])

        time_text.set_text('')
        energy_text.set_text('')
        return patch, points, time_text, energy_text

    def animate(i):
        """perform animation step"""
        global pendulum_qs
        pendulumsim.step(dt)
        #print(pendulumsim.dqdt2)
        positions = pendulumsim.positions()
        #line.set_data(positions[:,0], positions[:,1])
        patch.set_xy(positions)
        points.set_data(positions[:,0], positions[:,1])
        time_text.set_text('time = %.1f' % pendulumsim.time_elapsed)
        energy_text.set_text('energy = %.3f J' % sum(pendulumsim.energy(dt)))


        # box_qs = np.concatenate((box_qs, [boxsim.generalized_position()]))
        # line.set_data(box_qs[:i,0],box_qs[:i,1])
        # line.set_3d_properties(box_qs[:i, 2])

        return patch, points, time_text, energy_text


    interval = 1000/500
    print("Saving")
    ani = animation.FuncAnimation(fig, animate, frames=300,
                                  interval=interval, blit=True, init_func=init)#.save('double pendulum first try.gif', writer='imagemagick')


    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    #ani.save('auto_enc_opt.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()
    

    return ani

if __name__ == "__main__":
    path = "models/pendulum 6e-05 aug 28.h5"
    simulate(path)