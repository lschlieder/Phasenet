import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Lambda
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D

class DeepCGHBlock(tf.keras.layers.Layer):

    def __init__(self, input_shape, filters=[128, 256, 512], interleaving_factor=2):
        super(DeepCGHBlock,self).__init__()
        self.filters = filters
        self.interleaving_factor = interleaving_factor
        self.inputshape = input_shape
        self.pad = self.find_padding()



    def find_padding(self):
        newshape = tf.math.ceil(self.inputshape[0] / 2**3) * 2**3
        dif = newshape - self.inputshape[0]
        pad0 = int((dif/2).numpy())

        newshape = tf.math.ceil(self.inputshape[1] / 2 ** 3) * 2**3
        dif = newshape - self.inputshape[1]
        pad1 = int((dif/2).numpy())
        return pad0, pad1

    def get_complex_field(self, amplitude_and_phase):
        amplitude = amplitude_and_phase[0]
        phase = amplitude_and_phase[1]
        field = tf.cast(amplitude, dtype=tf.complex64) * tf.math.exp(1j * tf.cast(phase, dtype=tf.complex64))
        return field

    def interleave(self, x):
        if self.interleaving_factor is None:
            return x
        return tf.nn.space_to_depth(input=x,
                                    block_size=self.interleaving_factor,
                                    data_format='NHWC')

    def deinterleave(self, x):
        if self.interleaving_factor is None:
            return x
        return tf.nn.depth_to_space(input=x,
                                    block_size=self.interleaving_factor,
                                    data_format='NHWC')

    def cbn(self, ten, n_kernels, act_func):
        x1 = Conv2D(n_kernels, (3, 3), activation=act_func, padding='same')(ten)
        x1 = BatchNormalization()(x1)
        x1 = Conv2D(n_kernels, (3, 3), activation=act_func, padding='same')(x1)
        x1 = BatchNormalization()(x1)
        return x1

    def cc(self, ten, n_kernels, act_func):
        x1 = Conv2D(n_kernels, (3, 3), activation=act_func, padding='same')(ten)
        x1 = Conv2D(n_kernels, (3, 3), activation=act_func, padding='same')(x1)
        return x1

    def call(self, inp, **kwargs):
        x1_0 = ZeroPadding2D(self.pad)(inp)
        act_func = 'relu'
        x1_1 = Lambda(self.interleave, name='Interleave')(x1_0)
        # Block 1
        x1 = self.cbn(x1_1, self.filters[0], act_func)
        x2 = MaxPooling2D((2, 2), padding='same')(x1)
        # Block 2
        x2 = self.cbn(x2, self.filters[1], act_func)
        encoded = MaxPooling2D((2, 2), padding='same')(x2)
        # Bottleneck
        encoded = self.cc(encoded, self.filters[2], act_func)
        #
        x3 = UpSampling2D(2)(encoded)
        x3 = Concatenate()([x3, x2])
        x3 = self.cc(x3, self.filters[1], act_func)
        #
        x4 = UpSampling2D(2)(x3)
        x4 = Concatenate()([x4, x1])
        x4 = self.cc(x4, self.filters[0], act_func)
        #
        x4 = self.cc(x4, self.filters[1], act_func)
        x4 = Concatenate()([x4, x1_1])
        #
        amp_0_ = Conv2D(self.interleaving_factor ** 2, (3, 3), activation='relu', padding='same')(x4)
        amp_0 = Lambda(self.deinterleave, name='amp_0')(amp_0_)
        phi_0_ = Conv2D(self.interleaving_factor ** 2, (3, 3), activation=None, padding='same')(x4)
        phi_0 = Lambda(self.deinterleave, name='phi_0')(phi_0_)
        # Assemble complex field in image plane (not in the hologram plane, need to propagate in subsequent layer!)
        complex_field_image_plane = Lambda(self.get_complex_field)([amp_0, phi_0])
        cropped_complex_field = Cropping2D(self.pad)(complex_field_image_plane)
        return cropped_complex_field