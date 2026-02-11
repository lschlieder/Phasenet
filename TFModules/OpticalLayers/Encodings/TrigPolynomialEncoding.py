import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class TrigPolynomialEncoding(OpticalLayer):
    def __init__(self, num_coefficients = 4, output_img_size = (100,100), **kwargs):
        super( TrigPolynomialEncoding, self).__init__(**kwargs)

        self.output_img_size = output_img_size
        self.num_coefficients = num_coefficients


    
    def call(self, input, **kwargs):

        #input_resized = tf.image.resize(input, , )
        assert( input.shape[1]*self.num_coefficients <= self.output_img_size[0])
        assert(input.shape[2]*self.num_coefficients <= self.output_img_size[1])

        abs = tf.math.abs(input)
        phase = tf.math.angle(input)

        abs = abs/tf.reshape(tf.math.reduce_max(abs, axis = (1,2,3)),(tf.shape(abs)[0],1,1,1))

        n = 1
        col_images = []
        for x in range(0,self.num_coefficients):

            row_images = []
            for y in range(0,self.num_coefficients):
                img = tf.complex(abs,0.0)*tf.math.exp(1j*np.pi*n*tf.complex((abs),0.0))
                n = n+1
                row_images.append( img)

            row_image = tf.concat( row_images, axis = 1)
            col_images.append(row_image)
        image = tf.concat(col_images, axis = 2)


        
        #output = tf.complex(abs,0.0)*tf.math.exp(1j*np.pi*tf.complex((abs),0.0))
        #print(image.shape)
        return image
    
    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.output_img_size[0], self.output_img_size[1], 1]

        
        
        
        
        #phase = tf.math.angle(input)

