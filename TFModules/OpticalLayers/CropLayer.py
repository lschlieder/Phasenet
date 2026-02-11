import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class CropLayer(OpticalLayer):
    def __init__(self,image_size, L, padding, **kwargs):
        '''
        Crops an image that has been padded by the padding layer back to image size
        image_size: input image size in pixels.
        L: Size of the resulting image in m
        padding: size of the padding area in m
        '''
        super(CropLayer,self).__init__(**kwargs)
        self.image_size = np.array(image_size)
        self.L = np.array(L)
        self.padding = np.array(padding)
        assert( self.image_size.size == 1 or self.image_size.size == 2 )
        assert( self.padding.size == 1 or self.padding.size == 2 )
        assert( self.L.size == 1 or self.L.size == 2 )

        if self.image_size.size == 1:
            self.image_size = np.array([image_size,image_size])
        if self.L.size == 1:
            self.L = np.array([L,L])
        if self.padding.size == 1:
            self.padding = np.array([padding,padding])


        self.dx = (self.L+ self.padding*2)/self.image_size

        self.padding_pixel = np.round(self.padding/self.dx,0).astype('int32')
        self.image_size_cropped = image_size - self.padding_pixel*2


    def build(self,input_shape):

        return

    def call(self,input, **kwargs):
        assert( input.shape[1] == self.image_size_cropped[0]+self.padding_pixel[0]*2)
        assert( input.shape[2] == self.image_size_cropped[1]+self.padding_pixel[1]*2)

        img = tf.slice( input, [0,self.padding_pixel[0], self.padding_pixel[1], 0], [-1, self.image_size_cropped[0], self.image_size_cropped[1], input.shape[3]])
        return img

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.image_size_cropped[0] , self.image_size_cropped[1],input_shape[3]]

    def get_config(self):
        temp = {
            "image_size": self.image_size,
            "L": self.L,
            "padding": self.padding
        }
        return temp

    @classmethod
    def from_config(cls, config):
        return cls(**config)



