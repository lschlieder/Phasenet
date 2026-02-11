import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PhaseplateNetwork.TFModules.PropagationLayer import PropagationLayer

import matplotlib.pyplot as plt

epsilon = 1e-10

class ForwardModel(tf.keras.Model):

    def __init__(self, z, L, N, transducer_radius = 25, padding = 50, f0 = 2.00e6, cM = 1484, regularization = None, reg_const = 0.00001):
        super(ForwardModel, self).__init__()
        #self.holo = Hologram()
        #self.single_layer = create_single_layer()
        self.L = L
        self.N = N
        self.padding = padding
        self.z = z
        dx = L / N

        self.regularizer = regularization
        self.reg_const = reg_const

        X,Y = np.mgrid[-L/2:L/2:dx, -L/2:L/2:dx]
        amplitude = np.zeros_like(X)
        amplitude[X**2 + Y**2 < transducer_radius**2] = 1.0
        self.amplitude = np.reshape(amplitude, (amplitude.shape[0],amplitude.shape[1],1))

        #plt.imshow(amplitude)
        #plt.show()

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(N*N, activation = 'relu')
        self.conv1 = tf.keras.layers.Conv2D(20, (5, 5), padding='same', activation='relu',use_bias = False)
        self.conv2 = tf.keras.layers.Conv2D(30, (5, 5), padding='same', activation='relu', use_bias = False)
        self.conv3 = tf.keras.layers.Conv2D(40, (5,5), padding = 'same', activation = 'relu', use_bias = False)
        self.conv4 = tf.keras.layers.Conv2D(1, (5,5), padding = 'same', use_bias = False)

        self.flatten = tf.keras.layers.Flatten()
        self.dense2 = tf.keras.layers.Dense(N*N*4, activation = 'relu')
        self.dense3 = tf.keras.layers.Dense(N*N)


        #self.regularization_weights.append(dense1.weights[0])


        '''

        self.flatten_2 = tf.keras.layers.Flatten()
        self.dense1_2 = tf.keras.layers.Dense(N*N, activation = 'relu')
        self.conv1_2 = tf.keras.layers.Conv2D(20, (5, 5), padding='same', activation='relu',use_bias = False)
        self.conv2_2 = tf.keras.layers.Conv2D(30, (5, 5), padding='same', activation='relu', use_bias = False)
        self.conv3_2 = tf.keras.layers.Conv2D(40, (5,5), padding = 'same', activation = 'relu', use_bias = False)
        self.conv4_2 = tf.keras.layers.Conv2D(1, (5,5), padding = 'same', use_bias = False)

        self.flatten_2 = tf.keras.layers.Flatten()
        self.dense2_2 = tf.keras.layers.Dense(N*N*4, activation = 'relu')
        self.dense3_2 = tf.keras.layers.Dense(N*N)
        '''

        self.propagation_layers = []
        for z_i in z:
            self.propagation_layers.append(PropagationLayer(z_i, N, L, padding, f0, cM))

    def build(self, input_shape):
        print(self.dense1.weights)
        #self.regularization_weights = [self.dense1.weights[0], self.conv1.weights, self.conv2.weights, self.conv3.weights, self.conv4.weights, self.dense2.weights[0], self.dense3.weights[0]]

    def call(self, inputs, **kwargs):
        img = inputs

        #print(img.shape)
        #print(self.flatten(img).shape)
        #print(self.dense1(self.flatten(img)).shape)
        dense_out = tf.reshape( self.dense1(self.flatten(img)), (-1, self.N,self.N, 1))
        #print(dense_out.shape)
        u = self.conv4(self.conv3(self.conv2(self.conv1(dense_out))))
        u = tf.reshape(self.dense3(self.dense2(self.flatten(u))),(-1,self.N,self.N,1))

        #dense_out_2 = tf.reshape( self.dense1_2( self.flatten_2(u)), (-1, self.N, self.N, 1))
        #u_2 = self.conv4_2(self.conv3_2(self.conv2_2(self.conv1_2(dense_out_2))))
        #u2 = tf.reshape(self.dense3_2( self.dense2_2(self.flatten(u_2))), (-1, self.N, self.N, 1))
        #u = u2+u

        u = (self.amplitude + epsilon) * tf.exp(1j * tf.cast(u, dtype=tf.complex64))

        prop_u = []
        for layer in self.propagation_layers:
            #print(u.shape)
            prop_u.append(layer(u))
        res = tf.concat(prop_u, axis = 3)

        self.u_array = [u]


        if not self.regularizer == None:
            self.regularization_weights = [self.dense1.weights[0], self.dense2.weights[0],
                                       self.dense3.weights[0],self.conv1.weights[0], self.conv2.weights[0], self.conv3.weights[0], self.conv4.weights[0]]
            #self.convolutional_regularization_weights = []
            self.add_loss(self.reg_const*self.regularizer(self.regularization_weights))
        return res, [u]



    def get_propagation(self):
        return self.u_array