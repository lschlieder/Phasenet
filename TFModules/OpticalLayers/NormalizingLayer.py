import tensorflow as tf
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class NormalizingLayer(OpticalLayer):
    '''
    Layer with learnable bias and variance that can be used to learn a normalization for a layer
    '''
    def __init__(self,**kwargs):
        super(NormalizingLayer, self).__init__(**kwargs)
        self.bias =  tf.Variable(0.0), 
        self.variance = tf.Variable(1.0)

    def build(self, input_shape):

        self.trainable_weights = [self.bias, self.variance]
        self.constraints[tf.keras.constraints.NonNeg(), tf.keras.constraints.NonNeg()]
    
    def call(self, input):
        output = (input-self.bias)/self.variance
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


    def get_image_variables(self):
        return
