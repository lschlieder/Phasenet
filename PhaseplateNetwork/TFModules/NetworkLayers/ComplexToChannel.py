import tensorflow as tf

class ComplexToChannel(tf.keras.layers.Layer):
    def __init__(self, mode = 'cartesian', **kwargs):
        super(ComplexToChannel,self).__init__(**kwargs)
        self.mode = mode
        #self.channel_dim = channel_dim
        if mode not in ['cartesian', 'radial']:
            raise ValueError("self.mode had must be in [radial, cartesian]")

    @tf.function
    def call(self, input):
        if self.mode == 'cartesian':
            a = tf.math.real(input)
            b = tf.math.imag(input)
        elif self.mode =='radial':
            a = tf.math.abs(input)
            b = tf.math.angle(input)
        else:
            raise ValueError("self.mode had must be in [radial, cartesian]")
        return tf.concat([a,b],3)


