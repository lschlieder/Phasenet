import tensorflow as tf
epsilon = 0.0000000000000001

class SaturableAbsorbtionLayer(tf.keras.layers.Layer):
    '''
        Saturable absorbtion Layer. Implements
        res = exp(- (a0/2)/(1 + Ep**2)) * Ep
        which is relu like
    '''
    def __init__(self, alpha_0 = 10, **kwargs):
        self.a0 = alpha_0

    def call(self, inp):
        abs = tf.math.abs(inp)
        angle = tf.math.angle(inp)
        res = tf.math.exp(- ( self.a0/2)/(1 + abs**2)) * abs
        return tf.complex(abs+ epsilon,0.0) * tf.math.exp(tf.complex(0.0, angle))