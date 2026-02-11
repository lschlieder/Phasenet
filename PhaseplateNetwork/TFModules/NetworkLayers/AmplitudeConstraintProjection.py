import tensorflow as tf

epsilon = 1e-10

class AmplitudeConstraintProjection(tf.keras.layers.Layer):
    def __init__(self):
        super(AmplitudeConstraintProjection,self).__init__()

    @tf.function
    def call(self, input, wanted_amplitude):
        return tf.complex((tf.math.abs(wanted_amplitude) + epsilon),0.0) * tf.exp(1j * tf.cast(tf.math.angle(input), dtype=tf.complex64))
