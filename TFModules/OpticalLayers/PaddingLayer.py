import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class PaddingLayer(OpticalLayer):
    def __init__(self,image_size, L, padding,constant = 0.0, **kwargs):
        '''
        Pads an image of size L up to a size padding on both sides of each dimension
        image_size: the 1 or 2 dimensional image size
        L: size of the incoming image in meters
        padding: padding to be applied to both!! sides in meters
        '''
        super(PaddingLayer,self).__init__(**kwargs)


        self.image_size = np.array(image_size)
        self.L = np.array(L)
        self.padding = np.array(padding)
        assert( self.image_size.size == 1 or self.image_size.size == 2 )
        assert( self.padding.size == 1 or self.padding.size == 2 )
        assert( self.L.size == 1 or self.L.size == 2 )
        assert( np.array(constant).size == 1)

        if self.image_size.size == 1:
            self.image_size = np.array([image_size,image_size])
        if self.L.size == 1:
            self.L = np.array([L,L])
        if self.padding.size == 1:
            self.padding = np.array([padding,padding])


        self.dx = self.L/self.image_size
        self.paddings_pixel =np.round(self.padding/self.dx).astype('int32')

        self.paddings = tf.constant([[0,0],[self.paddings_pixel[0], self.paddings_pixel[0]],[self.paddings_pixel[1], self.paddings_pixel[1]], [0,0]])
        self.constant = constant

    def call(self,input, **kwargs):
        #print(input.shape)
        #print(self.image_size)
        #print("shapes test")
        assert( input.shape[1] == self.image_size[0])
        assert( input.shape[2] == self.image_size[1])
        #img = tf.image.pad_to_bounding_box(input, self.padding_pixel[0], self.padding_pixel[1], self.padding_pixel[0]*2 + self.image_size[0], self.padding_pixel[1]*2 + self.image_size[1])
        img = tf.pad( input, self.paddings,"CONSTANT", constant_values = self.constant)
        return img

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.image_size[0] + 2*self.paddings_pixel[0], self.image_size[1] + 2*self.paddings_pixel[1],input_shape[3]]


    def get_config(self):
        temp = {
            "image_size": self.image_size,
            "L": self.L,
            "padding": self.padding,
            "constant": self.constant
        }
        return temp

    @classmethod
    def from_config(cls, config):
        return cls(**config)



