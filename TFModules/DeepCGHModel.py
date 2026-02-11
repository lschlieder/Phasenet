from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Input, Concatenate, Lambda
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from PhaseplateNetwork.TFModules.PropagationLayer import PropagationLayer
from PhaseplateNetwork.TFModules.OpticalLayers.WavePropagation import WavePropagation
import tensorflow as tf
import numpy as np


class cbn(tf.keras.layers.Layer):
    def __init__(self, n_kernels=10, act_func='linear'):
        super(cbn, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(n_kernels, (3, 3), activation=act_func, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(n_kernels, (3, 3), activation=act_func, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inp):
        x1 = self.bn1(self.conv1(inp))
        x1 = self.bn2(self.conv2(x1))
        return x1


class cc(tf.keras.layers.Layer):
    def __init__(self, n_kernels=10, act_func='linear'):
        super(cc, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(n_kernels, (3, 3), activation=act_func, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(n_kernels, (3, 3), activation=act_func, padding='same')

    def call(self, inp):
        return self.conv2(self.conv1(inp))

class DeepCGHModel(tf.keras.Model):



    def __init__(self,z, L, N, transducer_radius=25, padding=50, f0=2.00e6, cM=1484, int_factor = 4):
        super(DeepCGHModel, self).__init__()

        self.IF = int_factor
        self.L = L
        self.N = N
        self.padding = padding
        self.z = tf.cast(z, dtype=tf.float32)
        self.transducer_radius = transducer_radius
        self.f0 = f0
        self.cM = cM
        dx = L / N
        X, Y = np.mgrid[-L / 2:L / 2:dx, -L / 2:L / 2:dx]

        act_func = 'relu'


        amplitude = np.zeros_like(X)
        amplitude[X ** 2 + Y ** 2 < transducer_radius ** 2] = 1.0
        self.amplitude = np.reshape(amplitude, (amplitude.shape[0], amplitude.shape[1], 1))

        n_kernels = [128,256,512]
        self.cbn1 = cbn(n_kernels[0], act_func)
        self.maxp1 = tf.keras.layers.MaxPooling2D((2,2), padding = 'same')
        self.cbn2 = cbn(n_kernels[1], act_func)
        self.maxp2 = tf.keras.layers.MaxPooling2D((2,2), padding = 'same')
        self.cc1 = cc(n_kernels[2], act_func)
        self.ups1 = tf.keras.layers.UpSampling2D(2)
        self.conc1 = tf.keras.layers.Concatenate()
        self.cc2 = cc(n_kernels[1], act_func)
        self.ups2 = tf.keras.layers.UpSampling2D(2)
        self.conc2 = tf.keras.layers.Concatenate()
        self.cc3 = cc(n_kernels[0], act_func)
        self.cc4 = cc(n_kernels[1],act_func)
        self.conc3 = tf.keras.layers.Concatenate()

        self.conv1 = tf.keras.layers.Conv2D(self.IF**2, (3,3), activation = None, padding = 'same')
        self.conv2 = tf.keras.layers.Conv2D(self.IF**2, (3,3), activation ='relu', padding = 'same')

        self.wave_propagation = WavePropagation(z, N, L, padding=padding, f0=f0, cM=cM)
        self.wave_propagation_backwards = WavePropagation(z[0], N, L , padding = padding, f0=f0, cM = cM)


    def call(self, inp):
        x1_1 = tf.nn.space_to_depth(input=inp,block_size=self.IF,data_format='NHWC')
        x1 = self.cbn1(x1_1)
        x2 = self.maxp1(x1)
        x2 = self.cbn2(x2)
        encoded = self.maxp2(x2)
        encoded = self.cc1(encoded)
        x3 = self.ups1(encoded)
        #print(inp.shape)
        #print(x1_1.shape)
        #print(x1.shape)
        #print(x2.shape)
        #print(x3.shape)
        #print(x2.shape)
        x3 = self.conc1([x3,x2])
        x3 = self.cc2(x3)

        x4 = self.ups2(x3)
        x4 = self.conc2([x4,x1])
        x4 = self.cc3(x4)

        x4 = self.cc4(x4)
        x4 = self.conc3([x4,x1_1])

        phi_0 = self.conv1(x4)
        phi_0 = tf.nn.depth_to_space(input = phi_0,block_size = self.IF,data_format = 'NHWC')

        amp_0 = self.conv2(x4)
        amp_0 = tf.nn.depth_to_space(input=amp_0, block_size=self.IF, data_format='NHWC')

        image_plane = tf.cast(amp_0+0.00000000001,dtype = tf.complex64)*tf.math.exp(1j*tf.cast(phi_0, dtype = tf.complex64))

        #print(image_plane.shape)

        #holo_plane = self.wave_propagation.inverse_call(image_plane)
        holo_plane = self.wave_propagation_backwards.inverse_call(image_plane)

        plate_angle = tf.math.angle(holo_plane)
        plate_amp = self.amplitude

        holo_plane_amp = tf.cast(plate_amp+0.000000001,dtype = tf.complex64) * tf.math.exp( 1j*tf.cast(plate_angle,dtype = tf.complex64))

        #print(holo_plane_amp.shape)

        propagated_image = self.wave_propagation.call(holo_plane_amp)
        return propagated_image, [holo_plane_amp]




