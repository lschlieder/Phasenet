import tensorflow as tf
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
import numpy as np

class AmplitudeToPhase(OpticalLayer):
    def __init__(self, max = 1.0, **kwargs):
        super(AmplitudeToPhase,self).__init__(**kwargs)
        self.max = max

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2],input_shape[3]]

    def call(self, input):
        phase = input/self.max * np.pi*2
        return tf.complex(tf.ones_like(input), 0.0) * tf.math.exp(1j * tf.complex(phase,0.0))