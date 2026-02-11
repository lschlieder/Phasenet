import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PhaseplateNetwork.TFModules.PropagationLayer import PropagationLayer

epsilon = 1e-10

class Dblock(tf.keras.Model):
    def __init__(self,*kargs):
        super(Dblock,self).__init__()
        self.b_norm1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(4,3, 2,padding = 'same', activation='relu')
        self.b_norm2= tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(4,3, padding = 'same', activation = 'relu')

    def call(self, inputs, training=None, mask=None):
        out = self.b_norm2(self.conv2(self.b_norm1(self.conv1(inputs))))
        return out

class Ublock(tf.keras.Model):
    def __init__(self, *kargs):
        super(Ublock,self).__init__()
        self.b_norm1 = tf.keras.layers.BatchNormalization()
        self.up_conv1 = tf.keras.layers.Conv2DTranspose(1,3,2, padding = 'same', activation = 'relu')
        self.b_norm2 = tf.keras.layers.BatchNormalization()
        self.up_conv2 = tf.keras.layers.Conv2DTranspose(1,3,1, padding = 'same', activation = 'relu')

    def call(self, inputs, training=None, mask=None):
        out = self.b_norm2(self.up_conv2(self.b_norm1(self.up_conv1(inputs))))
        return out


class UNetModel(tf.keras.Model):



    def __init__(self, z, L, N, transducer_radius = 25, padding = 50, f0 = 2.00e6, cM = 1484, regularization = None, reg_const = 0.00001):
        super(UNetModel, self).__init__()
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

        self.dblock1 = Dblock()
        self.dblock2 = Dblock()
        self.dblock3 = Dblock()
        self.dblock4 = Dblock()

        self.conv1 = tf.keras.layers.Conv2D(1, 3, padding = 'same')
        self.conv2 = tf.keras.layers.Conv2D(1, 3, padding = 'same')
        self.conv3 = tf.keras.layers.Conv2D(1, 3, padding = 'same')
        self.conv4 = tf.keras.layers.Conv2D(1, 3, padding = 'same')

        self.flatten = tf.keras.layers.Flatten()

        def get_downscaled_2(inp):

            for i in range(0,4):
                inp = np.ceil(inp/2)
            return int(inp)
        #print(N//(16))
        #print(get_downscaled_2(N))
        self.dense1 = tf.keras.layers.Dense(get_downscaled_2(N)**2 * 4, activation ='relu')

        self.dense_inp = tf.keras.layers.Dense(N*N*4, activation = 'relu')
        self.conv5 = tf.keras.layers.Conv2D(1, 3,padding = 'same', activation = 'relu')

        self.ublock1 = Ublock()
        self.ublock2 = Ublock()
        self.ublock3 = Ublock()
        self.ublock4 = Ublock()

        self.last_conv = tf.keras.layers.Conv2D(1, 3, padding = 'same')
        #plt.imshow(amplitude)
        #plt.show()

        #self.flatten = tf.keras.layers.Flatten()
        #self.dense1 = tf.keras.layers.Dense(N*N, activation = 'relu')
        #self.conv1 = tf.keras.layers.Conv2D(50, (5, 5), padding='same', activation='relu',use_bias = False)
        #self.conv2 = tf.keras.layers.Conv2D(128, (5, 5), padding='same', activation='relu', use_bias = False)
        #self.conv3 = tf.keras.layers.Conv2D(128, (5,5), padding = 'same', activation = 'relu', use_bias = False)
        #self.conv4 = tf.keras.layers.Conv2D(6, (5,5), padding = 'same', use_bias = False)

        #self.flatten = tf.keras.layers.Flatten()
        #self.dense2 = tf.keras.layers.Dense(N*N, activation = 'relu')
        #self.dense3 = tf.keras.layers.Dense(N*N)


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
        def pad_up_to(t, max_in_dims, constant_values):
            s = tf.shape(t)
            paddings = [[0, m - s[i]] for (i, m) in enumerate(max_in_dims)]
            return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)
        img = inputs

        #print(img.shape)
        #print(self.flatten(img).shape)
        #print(self.dense1(self.flatten(img)).shape)
        dense_out = tf.reshape( self.dense_inp(self.flatten(img)), (-1, self.N,self.N, 4))
        #print(dense_out.shape)
        d1 = self.dblock1(dense_out)
        d2 = self.dblock2(d1)
        d3 = self.dblock3(d2)
        d4 = self.dblock4(d3)

        #uf = self.flatten(d4)
        #print(d1.shape)
        #print(d2.shape)
        #print(d3.shape)
        #print(d4.shape)
        #print(uf.shape)
        #up = tf.reshape(self.dense1(uf),(-1, d4.shape[1], d4.shape[2], 1))

        up = self.conv5(d4)


        u4 = tf.slice(self.ublock4(up),[0,0,0,0],[d3.shape[0], d3.shape[1], d3.shape[2], 1]) + self.conv4(d3)

        u3 = tf.slice(self.ublock3(u4),[0,0,0,0],[d2.shape[0],d2.shape[1], d2.shape[2], 1]) + self.conv3(d2)

        u2 = tf.slice(self.ublock2(u3),[0,0,0,0],[d1.shape[0], d1.shape[1], d1.shape[2], 1]) + self.conv2(d1)
        u1 = tf.slice(self.ublock1(u2),[0,0,0,0],[inputs.shape[0], inputs.shape[1], inputs.shape[2],1]) + self.conv1(inputs)

        u = self.last_conv(u1)
        #print(u.shape)


        #u = self.conv4(self.conv3(self.conv2(self.conv1(dense_out))))
        #u = tf.reshape(self.dense3(self.dense2(self.flatten(u))),(-1,self.N,self.N,1))

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