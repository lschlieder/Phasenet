import tensorflow as tf
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class IntensityLayer(OpticalLayer):
    def __init__(self, factor = (2,2),interpolation = "nearest", **kwargs):
        super(IntensityLayer,self).__init__(**kwargs)

    
    def call(self, input, **kwargs):
        return tf.math.abs(input)**2

    def compute_output_shape(self, input_shape):
        return input_shape