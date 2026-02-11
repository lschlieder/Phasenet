import tensorflow as tf
import numpy as np

epsilon = 1e-10


class TransducerConstraintProjection(tf.keras.layers.Layer):
    def __init__(self, L, N, transducer_radius=25):
        super(TransducerConstraintProjection,self).__init__()
        dx = L / N
        X, Y = np.mgrid[-L / 2:L / 2:dx, -L / 2:L / 2:dx]
        amplitude = np.zeros_like(X)
        amplitude[X ** 2 + Y ** 2 < transducer_radius ** 2] = 1.0
        self.amplitude = np.reshape(amplitude, (amplitude.shape[0], amplitude.shape[1], 1))

    @tf.function
    def call(self, input):
        return (self.amplitude + epsilon) * tf.exp(1j * tf.cast(input, dtype=tf.complex64))
