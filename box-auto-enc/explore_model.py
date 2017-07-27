import tensorflow as tf
sess = tf.Session()

from keras_learn import *
from keras import backend as K
K.set_session(sess)

print("Loading...")
model = load_model('models/decent 500000 samples.h5')
decoder_input = Input(shape=(3,))
decoder_middle = model.layers[2](decoder_input)
decoded = model.layers[3](decoder_middle)
decoder = Model(decoder_input, decoded)

encoder_input = Input(shape=(8,))
encoder_middle = model.layers[0](encoder_input)
encoded = model.layers[1](encoder_middle)
encoder = Model(encoder_input, encoded)

x_bounds = [-10, 10]
y_bounds = [-10, 10]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_xlim(x_bounds)
ax.set_ylim(y_bounds)
ax.set_aspect('equal')

boxes = decoder.predict(np.array([[j/10.0,0.3455168, 0/10.0] for i in range(10) for j in range(10)]))
draw_boxes_from_samples(ax, boxes, 'b')
print("Done.")

# real_boxes = generate_box_samples_fast(1)
# encoded = encoder.predict(real_boxes)
# reconstructed = model.predict(real_boxes)

# draw_boxes_from_samples(ax, real_boxes, 'r', linewidth=5)
# draw_boxes_from_samples(ax, reconstructed, 'b')

# print("Real: ", real_boxes)
# print("Encoded: ", encoded)
# print("Reconstructed (decoder): ", decoder.predict(encoded))
# print("Reconstructed (full model): ", reconstructed)

# good sample
# Real:  [[ 0.3233729   0.91341918  0.25566781  0.83982557  0.18207419  0.90753066
#    0.24977929  0.98112428]]
# Encoded:  [[ 0.14605495  0.1455168   0.73541135]]
    
plt.show()