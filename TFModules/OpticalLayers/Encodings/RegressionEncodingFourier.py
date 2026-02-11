import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
from PhaseplateNetwork.utils.ImageUtils import get_gaussian_beam_image
from PhaseplateNetwork.utils.ImageUtils import create_hexagonal_image_stack

#import matplotlib.pyplot as plt

class RegressionEncodingFourier(OpticalLayer):
    '''
    Creates a gaussian beam with fourier_components phase regions, that contain phase inputs that are modified with increasing frequency.
    2 pi * i * x
    params:
    image_size: size of the image im [m]
    fourier_components: number of the first n fourier components 
    image_pixels: number of image output pixels (will be transformed to a tuple if given an int)
    sigma: sigma of the output gaussian function 
    '''
    def __init__(self, image_size, fourier_components = 10, image_pixels = 112, sigma = 0.5, **kwargs):
        super(RegressionEncodingFourier,self).__init__(**kwargs)

        self.image_size_inp = image_size
        self.image_pixels_inp = image_pixels
        self.sigma_inp = sigma
        self.fourier_components = fourier_components
        if isinstance(image_size, float):
            image_size = (image_size, image_size)

        assert( not isinstance(image_pixels, float))
        if isinstance(image_pixels, int):
            self.image_pixels = (image_pixels, image_pixels)
            

        self.gaussian_image = tf.cast(get_gaussian_beam_image(image_size, sigma, image_pixels = self.image_pixels), tf.float32)

        self.image_stack = create_hexagonal_image_stack(fourier_components, self.image_pixels, image_size[0], image_size[0]/np.sqrt(10*fourier_components)) 
        self.image_stack = tf.cast(self.image_stack, tf.float32)


    def call(self,input, **kwargs):
        assert(len(input.shape) == 2)
        assert(input.shape[1] == 1)
        #print(self.gaussian.dtype)
        input = tf.cast(input, tf.float32)

        out_abs = tf.reshape(tf.complex(self.gaussian_image, 0.0),(1,self.image_pixels[0],self.image_pixels[1],1))# * tf.math.exp( tf.complex(0.0, (2*np.pi*tf.reshape(tf.math.abs(input), (-1,1,1,1)))) )
        out_phase = tf.zeros((1, self.image_pixels[0], self.image_pixels[1],1))
        for i in range(0,self.fourier_components):
            # plt.figure()

            out_i =  2*np.pi*i*tf.reshape(tf.math.abs(input), (-1,1,1,1))
            #print(out_i)
            #print(out_phase.shape)
            #print(self.image_stack.dtype)
            #plt.imshow(out_i* self.image_stack[:,:,i])
            additional_phase = (out_i * tf.expand_dims(tf.expand_dims(self.image_stack[:,:,i], axis = 0), axis = 3))
            #print(additional_phase.shape)
            out_phase = out_phase +  additional_phase
            #plt.figure()
            #plt.imshow(additional_phase[0,:,:,0])

        #plt.imshow(out_phase[0,:,:,0])
        #plt.colorbar()
        out = out_abs * tf.math.exp( tf.complex( 0.0, out_phase))

        #out_top = tf.reshape(tf.complex(self.gaussian_image, 0.0),(1,self.image_pixels[0],self.image_pixels[1],1)) * tf.math.exp( tf.complex(0.0, (2*np.pi*tf.reshape(tf.math.abs(input), (-1,1,1,1)))) )

        #out_bottom = tf.reshape(tf.complex(self.gaussian_image, 0.0),(1,self.image_pixels[0],self.image_pixels[1],1)) * tf.math.exp( tf.complex(0.0, (2*np.pi*tf.reshape(tf.math.abs(tf.ones_like(input)), (-1,1,1,1)) )) )

        #half = int( np.floor( self.image_pixels[0]/2))
        #out = tf.concat( (out_top[:,:half,:,:], out_bottom[:,half:,:,:]), axis = 1)



        #out = tf.transpose(out, (0,2,3,1))
        return out 
    

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.image_pixels, self.image_pixels, 1]

    def get_config(self):
        temp = {
            'image_size': self.image_size_inp,
            'image_pixels': self.image_pixels_inp,
            'sigma': self.sigma_inp
        }
        return temp

