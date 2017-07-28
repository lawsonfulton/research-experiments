from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

import tensorflow as tf
sess = tf.Session()

from keras_learn import *
from keras import backend as K
K.set_session(sess)

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(1,2,1)
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.set_aspect('equal')

ax3d = fig.add_subplot(1,2,2, projection='3d')
ax3d.set_xlim([0,1])
ax3d.set_ylim([0,1])
ax3d.set_zlim([0,1])

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

encoder, decoder = get_autoencoder('models/decent 500000 samples.h5')

n_samples = 10
r = 6
thetas = np.zeros(n_samples)
offsets = np.array([[r * math.sin(theta), r * math.cos(theta)] for theta in np.linspace(0.0, 2*math.pi, num=n_samples)])

real_boxes = generate_box_samples_fast(offsets=offsets, thetas=thetas)
#boxes = decoder.predict(np.array([[j/10.0,0.3455168, 0/10.0] for i in range(10) for j in range(10)]))
encoded_boxes = encoder.predict(real_boxes)
decoded_boxes = decoder.predict(encoded_boxes)

# draw_boxes_from_samples(ax, real_boxes, 'r', linewidth=5)
# draw_boxes_from_samples(ax, decoded_boxes, 'b')

# ax3d.scatter(xs=encoded_boxes[:,0], ys=encoded_boxes[:,1], zs=encoded_boxes[:,2])
    
# plt.show()


###
# def gen(n):
#     phi = 0
#     while phi < 2*np.pi:
#         yield np.array([np.cos(phi), np.sin(phi), phi])
#         phi += 2*np.pi/n

# def update(num, data, line):
#     line.set_data(data[:2, :num])
#     line.set_3d_properties(data[2, :num])

# N = 100
# data = np.array(list(gen(N))).T
# line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

# # Setting the axes properties
# ax.set_xlim3d([-1.0, 1.0])
# ax.set_xlabel('X')

# ax.set_ylim3d([-1.0, 1.0])
# ax.set_ylabel('Y')

# ax.set_zlim3d([0.0, 10.0])
# ax.set_zlabel('Z')

# ani = animation.FuncAnimation(fig, update, N, fargs=(data, line), interval=10000/N, blit=False)

####

line, = ax3d.plot(encoded_boxes[0:1,0], encoded_boxes[0:1,1], encoded_boxes[0:1,2])

def animate(i):
#     line.set_ydata(np.sin(x + i/10.0))  # update the data
    draw_boxes_from_samples(ax, [real_boxes[i]], 'r', linewidth=5)
    draw_boxes_from_samples(ax, [decoded_boxes[i]], 'b')
    
    line.set_data(encoded_boxes[:i,:2])
#     line.set_data(encoded_boxes[0:i,0], encoded_boxes[0:i,1])
    line.set_3d_properties(encoded_boxes[:i, 2])
#    ax3d.plot(encoded_boxes[i,0], encoded_boxes[i,1], encoded_boxes[i,2])
    #return line,


# # Init only required for blitting to give a clean slate.
# def init():
#     line.set_ydata(np.ma.array(x, mask=True))
#     return line,

print("animating")

anim = animation.FuncAnimation(fig, animate, frames=n_samples, interval=20, blit=False)#True)
print("loading video")
# HTML(anim.to_html5_video())
#anim.save('matplot003.gif', writer='imagemagick')
print("done")
plt.show()
