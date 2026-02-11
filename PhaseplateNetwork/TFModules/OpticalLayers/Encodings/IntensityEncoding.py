import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class IntensityEncoding(OpticalLayer):
    def __init__(self, **kwargs):
        super(IntensityEncoding,self).__init__(**kwargs)


    def call(self,input, **kwargs):
        abs = tf.math.abs(input)
        phase = tf.math.angle(input)

        abs = abs/tf.reshape(tf.math.reduce_max(abs, axis = (1,2,3)),(tf.shape(abs)[0],1,1,1))
        output = tf.complex(abs,0.0)*tf.math.exp(1j*tf.complex(tf.zeros_like(abs),0.0))

        return output

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2],1]

    def get_config(self):
        temp = {
        }
        return temp
