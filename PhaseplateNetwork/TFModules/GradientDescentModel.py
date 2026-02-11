import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PhaseplateNetwork.TFModules.PropagationLayer import PropagationLayer
from PhaseplateNetwork.Losses.Losses import mse, generator_loss


epsilon = 1e-10

@tf.function
def loss_fn(regularizer_output, input,output, alpha = 1.0):

    mse_n = mse(input, output)
    loss_from_disc = tf.math.reduce_mean(
        tf.keras.backend.binary_crossentropy(tf.ones_like(regularizer_output), regularizer_output, True))
    return mse_n + alpha * loss_from_disc

class GradientDescentModel(tf.keras.Model):
    def __init__(self, z, L, N, transducer_radius = 25, padding = 50, f0 = 2.00e6, cM = 1484, depth = 30, optimizer_grad = tf.keras.optimizers.SGD(1.0)):
        super(GradientDescentModel, self).__init__()
        self.depth = depth
        self.propagation_layers = []
        self.opt= optimizer_grad
        dx = L / N
        X,Y = np.mgrid[-L/2:L/2:dx, -L/2:L/2:dx]
        amplitude = np.zeros_like(X)
        amplitude[X**2 + Y**2 < transducer_radius**2] = 1.0
        self.amplitude = np.reshape(amplitude, (amplitude.shape[0],amplitude.shape[1],1)).astype('float32')
        self.propagation_layer = PropagationLayer(np.array(z), N, L, padding, f0, cM)
        #for z_i in z:
        #    self.propagation_layers.append(PropagationLayer(z_i, N, L, padding, f0, cM))

        #self.u_angle = tf.Variable(tf.random.uniform(shape = (inputs.shape[0], inputs.shape[1], inputs.shape[2],1),dtype = tf.float32), trainable = True)
        return

    def build(self, input_shape):
        self.u_angle =tf.Variable(tf.random.uniform(shape = (input_shape[0], input_shape[1], input_shape[2],1),dtype = tf.float32), trainable = True)


    def call(self, inputs, regularizer = None,  **kwargs):
        #print(inputs.shape)

        #self.u_angle.assign(tf.random.uniform(shape = (inputs.shape[0], inputs.shape[1], inputs.shape[2],1),dtype = tf.float32))
        self.u_angle.assign(tf.constant(0.0, shape = (inputs.shape[0], inputs.shape[1], inputs.shape[2],1), dtype = tf.float32))
        for i in range(0, self.depth):
            #print(u_angle)
            with tf.GradientTape() as g:
                g.watch(self.u_angle)
                u = tf.complex(self.amplitude + epsilon, 0.0) * tf.math.exp( tf.complex(0.0, self.u_angle))
                #prop_u = []
                #for layer in self.propagation_layers:
                    # print(u.shape)
                    #prop_u.append(layer(u))
                #res = tf.concat(prop_u, axis = 3)
                res = self.propagation_layer(u)
                if regularizer != None:
                    fake_output = regularizer(res)
                    loss = generator_loss( fake_output, inputs, res)
                else:
                    res = tf.math.abs(res)
                    loss = mse_loss(res, inputs)

            #print(res)
            #plt.imshow(res[0,:,:,0])
            #plt.show()
            #print(self.u_angle)
            gradient = g.gradient(loss, [self.u_angle])
            #print(gradient)
            self.opt.apply_gradients(zip( gradient, [self.u_angle]))

        #plt.imshow(self.u_angle[0,:,:,0])
        #plt.show()
        #input()
        u = tf.convert_to_tensor(self.u_angle)
        return res, [u]




