import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
import matplotlib.pyplot as plt

class JonesSaturableAbsorber(OpticalLayer):
    def __init__(self, OD = 1.0, I_saturation = 0.1, I_sat_trainable = False, **kwargs):
        super(JonesSaturableAbsorber,self).__init__(**kwargs)
        self. OD = OD
        self.I_saturation = I_saturation
        self.I_sat_trainable = I_sat_trainable
        self.I_sat = tf.Variable(I_saturation, trainable=I_sat_trainable)

    def call(self, Input):
        Intensity = tf.expand_dims(tf.reduce_sum(tf.math.abs(Input)**2, axis = 3),axis = 3)

        amp = tf.math.abs(Input) * tf.math.exp( -(self.OD/2)/(1+Intensity/self.I_sat))
        phase = tf.math.angle(Input)

        return tf.complex(amp,0.0) * tf.math.exp(1j* tf.complex(phase,0.0))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        temp = {
            "OpticalDensity": self.OD,
            "SaturationIntensity": self.I_saturation,
            "SaturationTrainable": self.I_sat_trainable
        }
        return temp

    @classmethod
    def from_config(cls, config):
        el = cls( OD = config['OpticalDensity'], I_saturation = config['SaturationIntensity'], I_sat_trainable= config['SaturationTrainable'])
        return el