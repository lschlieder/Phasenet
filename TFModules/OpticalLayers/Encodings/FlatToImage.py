import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class FlatToImage(OpticalLayer):
    def __init__(self, output_size = 112, **kwargs):
        super(FlatToImage,self).__init__(**kwargs)
        if isinstance(output_size,float) or isinstance(output_size,int):
            output_size = (output_size,output_size)

        self.output_size = output_size
        self.out_num = output_size[0]*output_size[1]

    def call(self,input, **kwargs):
        diff = self.out_num - input.shape[1]
        assert(diff >= 0)

        half_top = np.floor(diff/2).astype(int)
        half_bottom = np.ceil(diff/2).astype(int)
        out = tf.pad(input, [[0,0], [half_top, half_bottom]], constant_values = 0 )
        out = tf.reshape( out, (-1, self.output_size[0], self.output_size[1], 1) )
        return out

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.output_size[0], self.output_size[1],1]
    
    def get_propagation_fields(self, input):
        return []

    def get_config(self):
        temp = {
        }
        return temp
