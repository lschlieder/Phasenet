import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class CroppingLayerPixels(OpticalLayer):
    def __init__(self,image_size, new_image_size, **kwargs):
        super(CroppingLayerPixels,self).__init__(**kwargs)


        self.image_size = np.array(image_size)
        self.new_image_size = np.array(new_image_size)
        #self.padding = np.array(padding)
        assert( self.image_size.size == 1 or self.image_size.size == 2 )
        assert( self.new_image_size.size == 1 or self.new_image_size.size == 2 )

        if self.image_size.size == 1:
            self.image_size = np.array([image_size,image_size])
        if self.new_image_size.size == 1:
            self.new_image_size = np.array([new_image_size,new_image_size])

        

        assert( self.image_size[0] >= self.new_image_size[0])
        assert( self.image_size[1] >= self.new_image_size[1])

        image_diff = self.image_size - self.new_image_size

        crop_left = np.floor(image_diff/2).astype('int')
        crop_right = np.floor(image_diff/2).astype('int')
        #padding_left = np.floor(image_diff/2).astype('int')
        #padding_right = np.ceil(image_diff/2).astype('int')


        self.offset = crop_left
        #self.size = new_image_size
        #self.paddings = tf.constant([[0,0],[padding_left[0], padding_right[0]],[padding_left[1], padding_right[1]], [0,0]])
        #self.constant = constant

    def call(self,input):
        #print(input.shape)
        #print(self.image_size)
        #print("shapes test")
        assert( input.shape[1] == self.image_size[0])
        assert( input.shape[2] == self.image_size[1])

        #img = tf.image.pad_to_bounding_box(input, self.padding_pixel[0], self.padding_pixel[1], self.padding_pixel[0]*2 + self.image_size[0], self.padding_pixel[1]*2 + self.image_size[1])
        #img = tf.pad( input, self.paddings,"CONSTANT", constant_values = self.constant)
        img = tf.image.crop_to_bounding_box(input, self.offset[1], self.offset[0], self.new_image_size[1], self.new_image_size[0])
        return img

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.new_image_size[0], self.new_image_size[1], input_shape[3]]


    def get_config(self):
        temp = {
            "image_size": self.image_size,
            "new_image_size": self.new_image_size
        }
        return temp

    @classmethod
    def from_config(cls, config):
        return cls(**config)



