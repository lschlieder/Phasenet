import tensorflow as tf
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class SumLayer(OpticalLayer):
    def __init__(self, factor = (2,2),interpolation = "nearest", **kwargs):
        super(SumLayer,self).__init__(**kwargs)

    
    def call(self, input):
        res = tf.reduce_sum(input, axis = (1,2))

        return res

    def compute_output_shape(self, input_shape):
        if input_shape[3] == 1:
            return [input_shape[0], 1,input_shape[3]]
        else:
            return [input_shape[0],1,input_shape[3]]
