import tensorflow as tf
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
class AbsLayer(OpticalLayer):
    '''
    Layer with learnable bias and variance that can be used to learn a normalization for a layer
    '''
    def __init__(self,**kwargs):
        super(AbsLayer, self).__init__(**kwargs)
    
    def call(self, input):
        return tf.math.abs(input)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_image_variables(self):
        return