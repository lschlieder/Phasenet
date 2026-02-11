import numpy as np
import tensorflow as tf

# define the network core
class ODEModel(tf.keras.Model):
    def __init__(self, support_region, z, train_iter, test_iter):
        super(ODEModel, self).__init__()
        self.holo = Hologram()
        # init = tf.keras.initializers.Constant(value=(0.1 - 0.1*1j))

        self.conv1 = tf.keras.layers.Conv2D(2, (5, 5), padding='same', activation='relu')
        self.support_region = tf.cast(support_region, dtype=tf.float32)
        zeros = tf.constant(0.0, shape=self.support_region.shape)
        self.support_region = tf.complex(self.support_region, zeros)
        self.z = tf.cast(z, dtype=tf.float32)

        self.train_iter = train_iter
        self.test_iter = test_iter

    def set_images(self, images):
        self.images = tf.cast(images + 0 * 1j, dtype=tf.complex64)

    def call(self, inputs, **kwargs):
        t, u = inputs
        self.images = next(self.train_iter)
        proj = self.holo.get_projection_to_M(u, self.images, self.z, True)
        inp = tf.concat([u, proj], 3)
        inp_c = complex_to_channel(inp)
        u_new = self.conv1(inp_c)
        support_reg = tf.reshape(self.support_region, (1, u_new.shape[1], u_new.shape[2], 1))
        support_reg = tf.tile(support_reg, (u_new.shape[0], 1, 1, 1))
        u_new_c = channel_to_complex(u_new)
        u_new_c = u_new_c * support_reg
        return u_new_c


@tf.function
def compute_gradients_and_update_path(batch_y0, holo, z, batch_yN):
    with tf.GradientTape() as g:
        pred_y, y_points = neural_ode.forward(batch_y0, return_states="tf")
        pred_path = tf.stack(y_points)  # -> (batch_time, batch_size, 2)
        # print(np.shape(y_points))
        # print(np.shape(pred_path))
        pred_path = tf.reshape(pred_path,
                               (pred_path.shape[0], pred_path.shape[1], 1, pred_path.shape[2], pred_path.shape[3]))
        # pred_path = tf.transpose(pred_path, [1,0,2,3,4])
        images_predicted = holo.PropagateToPlaneTF(pred_path, z)
        # print('images_predicted shape: {}'.format(np.shape(images_predicted)))
        # print('batch_yN shape: {}'.format(batch_yN.shape))
        batch_yN = model.images
        print(batch_yN.shape)
        # wanted_pred = tf.reshape( batch_yN, (1, batch_yN.shape[0], batch_yN.shape[1],batch_yN.shape[2], batch_yN.shape[3] ))
        wanted_pred = tf.reshape(batch_yN, (1, batch, batch_yN.shape[1], batch_yN.shape[2], batch_yN.shape[3]))
        wanted_pred = tf.cast(tf.tile(wanted_pred, (pred_path.shape[0], 1, 1, 1, 1)), dtype=tf.float32)
        wanted_pred = tf.transpose(wanted_pred, [0, 1, 4, 2, 3])
        # print('wanted_pred shape: {}'.format(wanted_pred.shape))
        # print(np.shape(images_predicted))
        # print(np.shape(wanted_pred))
        # print(images_predicted)
        # for i in range(0,5):
        #   plt.figure()
        #  plt.imshow(tf.math.abs(images_predicted[i,0,1,:,:]).numpy())
        # plt.savefig('images_predicted_{}'.format(i))
        # plt.close()
        #  plt.figure()
        #  plt.imshow(wanted_pred.numpy()[i,0,1,:,:])
        #  plt.savefig('wanted_pred_{}'.format(i))
        #  plt.close()
        # plt.figure()
        # plt.imshow(images_predicted
        loss = tf.reduce_mean(tf.math.abs(images_predicted) - wanted_pred)

        # print(loss.numpy())
        # plt.imshow(tf.math.abs(images_predicted).numpy()[4,0,:,:])
        # plt.figure()
        # plt.imshow(np.angle(pred_y.numpy()))
        # plt.figure()
        # plt.imshow(wanted_pred[4,0,:,:])
        # plt.show()
        # loss = tf.reduce_mean(tf.abs(pred_path - wanted_pred), axis=1) # -> (batch_time, 2)
        # loss = tf.reduce_mean(loss, axis=0)

    # backpropagate through solver with tensorflow
    gradients = g.gradient(loss, model.weights)
    optimizer.apply_gradients(zip(gradients, model.weights))
    # print(pred_y.shape)
    return loss, pred_y, holo.PropagateToPlaneTF(pred_y, z, True), wanted_pred