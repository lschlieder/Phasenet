import tensorflow as tf

from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class ReluLayer(OpticalLayer):
    def __init__(self,bias = 0.1, **kwargs):
        super(ReluLayer, self).__init__(**kwargs)
        self.bias = bias

    def call(self, input):
        amp = tf.math.abs(input)
        angle = tf.math.angle(input)

        return tf.complex(tf.nn.relu(amp - self.bias),0.0) * tf.math.exp(1j*tf.complex(angle,0.0))

    def compute_output_shape(self, input_shape):
        return input_shape

