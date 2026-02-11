import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN

class JonesPropagation(DDNN):
    def __init__(self, z = 0.13, trainable_pixel = 112, plate_scale_factor = 2, propagation_size = 0.001792, propagation_pixel = 224, padding = 0.00175, frequency = 3.843e14, wavespeed = 3e8, **kwargs):
        super(IntensityInput,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed, **kwargs)
        input_shape = [None, self.propagation_pixel, self.propagation_pixel, 1]
        #input handling
        input_shape = self.append_element(self.get_padding_layer((input_shape[1],input_shape[2])), input_shape)
        input_shape = self.append_element(self.get_wave_propagation([z], input_shape[1]),input_shape)
        output_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)
        return
