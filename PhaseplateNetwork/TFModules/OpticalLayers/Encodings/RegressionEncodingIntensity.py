import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class RegressionEncodingIntensity(OpticalLayer):
    '''
    Bundle class for all jones polarization encodings
    '''
    def __init__(self, image_size = 112, sigma = 0.5, **kwargs):
        super(RegressionEncodingIntensity,self).__init__(**kwargs)
        self.image_size = image_size
        szIm = int(image_size/2)

        x, y = np.meshgrid(np.linspace(-1,1,szIm), np.linspace(-1,1,szIm))
        d = np.sqrt(x*x+y*y)
        sigma, mu = sigma, 0.0
        g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        self.sIm = szIm
        self.gaussian = tf.constant(g, dtype = tf.float32)


    def call(self,input, **kwargs):
        assert(len(input.shape) == 2)
        assert(input.shape[1] == 1)
        #print(self.gaussian.dtype)
        input = tf.cast(input, tf.float32)
        g11 = tf.reshape(tf.complex(self.gaussian, 0.0),(1,self.sIm, self.sIm, 1)) * tf.complex(tf.reshape( tf.ones_like(input), (-1,1,1,1)), 0.0)
        g21 = tf.reshape(tf.complex(self.gaussian, 0.0),(1,self.sIm, self.sIm, 1)) * tf.complex(tf.reshape( tf.ones_like(input), (-1,1,1,1)), 0.0)

        g12 = tf.reshape(tf.complex(self.gaussian, 0.0), (1,self.sIm, self.sIm, 1)) * tf.complex(tf.reshape(tf.math.abs(input), (-1,1,1,1)), 0.0)
        #g22 = g11 * tf.math.exp(tf.complex(0.0, 2*np.pi*tf.reshape(tf.math.abs(input), (-1,1,1,1))))
        g22 = tf.reshape(tf.complex(self.gaussian, 0.0), (1,self.sIm, self.sIm, 1)) * tf.complex(tf.reshape(tf.math.abs(input), (-1,1,1,1)), 0.0)

        top = tf.concat([g11,g12] , axis = 2)
        bottom = tf.concat([g21,g22], axis = 2)
        out = tf.concat([top,bottom], axis = 1)
        return out 
    

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.image_size, self.image_size, 1]

    def get_config(self):
        temp = {
            'image_size': self.image_size
        }
        return temp

