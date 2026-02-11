import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
from .CircularEncoding import CircularEncoding
from .LinearEncoding import LinearEncoding
from .LinearIntensityEncoding import LinearIntensityEncoding

class RegressionEncoding(OpticalLayer):
    '''
    Bundle class for all jones polarization encodings
    '''
    def __init__(self, image_size = 120, **kwargs):
        super(RegressionEncoding,self).__init__(**kwargs)

        self.image_size = image_size

        



    def call(self,input):
        assert(len(input.shape) == 2)
        assert(input.shape[1] == 1)
        

        y = tf.cast(tf.tile( tf.ones_like(tf.reshape(input, (-1,1,1,1))), (1,self.image_size, self.image_size, 1)), tf.complex64)
        #y  = tf.complex( tf.ones((1,self.image_size, self.image_size, 1)), 0.0) 

        half = int(self.image_size/2)

        amp = tf.complex( tf.ones((1,half*self.image_size, 1))*tf.reshape(tf.math.abs(input), (-1,  1, 1)), 0.0)

        phase = np.pi*2*tf.ones((1,half*self.image_size, 1))*tf.reshape(tf.math.abs(input), (-1,  1, 1))

        phase = tf.complex( tf.ones_like(phase), 0.0) * tf.math.exp(tf.complex(0.0,phase))
        #print(phase.shape)
        #print(amp.shape)
        #print(f"(1,{self.image_size},{self.image_size}, 1)")
        x = tf.reshape( tf.concat((amp,phase), axis = (1)), (-1, self.image_size, self.image_size, 1))

        out = tf.concat((x,y), axis = 3)
        return out 

        #return self.layer(input)

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.image_size, self.image_size, 2]

    def get_config(self):
        temp = {
            'image_size': self.image_size
        }
        return temp

