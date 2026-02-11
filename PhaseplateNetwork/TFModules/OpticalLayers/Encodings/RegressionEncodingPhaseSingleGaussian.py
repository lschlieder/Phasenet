import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
from PhaseplateNetwork.utils.ImageUtils import get_gaussian_beam_image

class RegressionEncodingPhaseSingleGaussian(OpticalLayer):
    '''
    Bundle class for all jones polarization encodings
    '''
    def __init__(self, image_size, image_pixels = 112, sigma = 0.5, **kwargs):
        super(RegressionEncodingPhaseSingleGaussian,self).__init__(**kwargs)


        self.image_pixels = image_pixels
        if isinstance(image_size, float):
            image_size = (image_size, image_size)

        assert( not isinstance(image_pixels, float))
        if isinstance(image_pixels, int):
            image_pixels = (image_pixels, image_pixels)

        self.gaussian_image = tf.cast(get_gaussian_beam_image(image_size, sigma, image_pixels = image_pixels), tf.float32)
        print(self.gaussian_image.shape)
        print(image_pixels)


    def call(self,input, **kwargs):
        assert(len(input.shape) == 2)
        assert(input.shape[1] == 1)
        #print(self.gaussian.dtype)
        input = tf.cast(input, tf.float32)
        out = tf.reshape(tf.complex(self.gaussian_image, 0.0),(1,self.image_pixels,self.image_pixels,1)) * tf.math.exp( tf.complex(0.0, (2*np.pi*tf.reshape(tf.math.abs(input), (-1,1,1,1)))) )

        #out = tf.transpose(out, (0,2,3,1))
        return out 
    

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.image_pixels, self.image_pixels, 1]

    def get_config(self):
        temp = {
            'image_size': self.image_size
        }
        return temp

