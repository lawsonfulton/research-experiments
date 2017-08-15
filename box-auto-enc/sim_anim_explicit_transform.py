import autograd.numpy as np
from autograd import grad, jacobian

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

import boxes
boxes.np = np

# q = np.array([0, 0, np.pi/4])
# x = boxes.box_from_vec(boxes.explicit_decode(q))
# jacxq = jacobian(boxes.explicit_decode)
# jac = jacxq(q)
# print(q)
# print(x)
# print(jac)

# x_bounds = [-10, 10]
# y_bounds = [-10, 10]
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111)
# ax.set_xlim(x_bounds)
# ax.set_ylim(y_bounds)
# ax.set_aspect('equal')

# boxes.draw_box(x, ax)
# plt.show()

# exit()
class BoxSim:
    def __init__(self):
        self.time_elapsed = 0.0

        self.world_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.world_force = np.array([0.0, -9.8, 0.0, -9.8, 0.0, -9.8, 0.0, -9.8])

        self.gen_pos = np.array([0.0, 0.0, 0.0])
        self.gen_vel = np.array([0.0, 0.0, 0.0])
        self.gen_acc = np.array([0.0, 0.0, 0.0])

        self.state = np.array([*self.gen_pos, *self.gen_vel])

        self.jac_x_wrt_q = jacobian(boxes.explicit_decode) # Preconstruct the jacobian function

    def mass_matrix(self, q):
        jac_matrix = self.jac_x_wrt_q(q)
        return np.dot(jac_matrix.transpose(), jac_matrix)

    def positions(self):
        return boxes.box_from_vec(boxes.explicit_decode(self.state[0:3]))

    # def energy(self):
    #     """compute the energy of the current state"""
    #     (L1, L2, M1, M2, G) = self.params

    #     x = np.cumsum([L1 * sin(self.state[0]),
    #                    L2 * sin(self.state[2])])
    #     y = np.cumsum([-L1 * cos(self.state[0]),
    #                    -L2 * cos(self.state[2])])
    #     vx = np.cumsum([L1 * self.state[1] * cos(self.state[0]),
    #                     L2 * self.state[3] * cos(self.state[2])])
    #     vy = np.cumsum([L1 * self.state[1] * sin(self.state[0]),
    #                     L2 * self.state[3] * sin(self.state[2])])

    #     U = G * (M1 * y[0] + M2 * y[1])
    #     K = 0.5 * (M1 * np.dot(vx, vx) + M2 * np.dot(vy, vy))

    #     return U + K

    def dstate_dt(self, state, t):
        """compute the derivative of the given state"""
        dsdt = np.zeros_like(state)

        q = state[0:3]
        dqdt = state[3:6]

        # mass_matrix = self.mass_matrix(q) # M(q)
        # mass_matrix_inv = np.linalg.inv(mass_matrix) # slow?

        #dvdt = mass_matrix_inv * self.jac_x_wrt_q(q) * mass_matrix * self.world_force
        jacxq = self.jac_x_wrt_q(q)
        dvdt = np.dot(self.world_force, jacxq)
        
        dsdt[0:3] = dqdt # dpos/dt = vel
        dsdt[3:6] = dvdt

        return dsdt

    def step(self, dt):
        """execute one time step of length dt and update state"""
        self.state = integrate.odeint(self.dstate_dt, self.state, [0, dt])[1]
        self.time_elapsed += dt


boxsim = BoxSim()
dt = 1.0 / 20

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure(figsize=(8,8))
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
    global boxsim, dt
    boxsim.step(dt)

    positions = boxsim.positions()
    #line.set_data(positions[:,0], positions[:,1])
    patch.set_xy(positions)
    points.set_data(positions[:,0], positions[:,1])
    time_text.set_text('time = %.1f' % boxsim.time_elapsed)
    # energy_text.set_text('energy = %.3f J' % boxsim.energy())
    return patch, points, time_text, energy_text

# choose the interval based on dt and the time to animate one step
from time import time
t0 = time()
#animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)
print (interval)
ani = animation.FuncAnimation(fig, animate, frames=300,
                              interval=interval, blit=True, init_func=init)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#ani.save('double_pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()