import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class IntensityPhaseEncoding(OpticalLayer):
    def __init__(self, output_size = (100,100), **kwargs):
        super(self, IntensityPhaseEncoding).__init__(**kwargs)
        self.output_size_x, self.output_size_y = output_size

        self.size_left_x = self.output_size //2 + self.output_size_x %  2
        self.size_left_y = self.output_Size //2 + self.output_size_y %2
        self.right_size_x = self.output_size//2
        self.right_size_y = self.output_size//2
    
    def call(self, input, **kwargs):

        #input_resized = tf.image.resize(input, , )
        abs = tf.math.abs(input)
        channels = self.input.shape[-1]
        amp = tf.complex(tf.image.resize(abs, size = (1,self.size_left_x, self.size_left_y)),0.0)
        phase = tf.math.exp(   tf.complex( 0.0, tf.image.resize(abs, size = (1,self.size_right_x, self.size_right_y)) ))

        ones_fill = tf.complex(tf.ones( shape = (1,self.size_right_x, self.size_left_y)), 0.0)
        ones_fill_2 = tf.complex(tf.ones( shape = (1,self.size_left_x, self.size_right_y)), 0.0) 

        output = self.concat( (amp, ones_fill), axis = 0)
        
        
        
        #phase = tf.math.angle(input)

