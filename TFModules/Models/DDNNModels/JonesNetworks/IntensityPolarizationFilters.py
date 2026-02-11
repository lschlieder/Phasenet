import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
import PhaseplateNetwork.TFModules.OpticalLayers.Encodings.PolarizationEncodings as PE
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN

class IntensityPolarizationFilters(DDNN):
    def __init__(self, axis = 'x', trainable_polarization = 'xy',block_num = 4, distance = 0.04, trainable_pixel = 120, plate_scale_factor = 1, propagation_size = 0.001792, propagation_pixel = 120, padding = 0.001, frequency = 3.843e14, wavespeed = 3e8, **kwargs):
        super(IntensityPolarizationFilters,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed, **kwargs)
        input_shape = [None, self.propagation_pixel, self.propagation_pixel, 1]
        #input handling
        input_shape = self.append_element(self.get_padding_layer((input_shape[1],input_shape[2])), input_shape)
        input_shape = self.append_element(PE.LinearIntensityEncoding('xy'), input_shape)
        #propagation and phaseplates
    
        for i in range(0,block_num):
            for j in range(0,2):
                input_shape = self.append_element(self.get_wave_propagation([distance] ,input_shape[1]), input_shape)
                input_shape = self.append_element(self.get_jones_phaseplate(False,True, trainable_polarization),input_shape)
                input_shape = self.append_element(self.get_wave_propagation([distance] ,input_shape[1]), input_shape)
            input_shape = self.append_element(OL.LinearPolarizationFilter(angle = np.pi/4), input_shape)
            for j in range(0,2):
                input_shape = self.append_element(self.get_wave_propagation([distance] ,input_shape[1]), input_shape)
                input_shape = self.append_element(self.get_jones_phaseplate(False,True, trainable_polarization),input_shape)
                input_shape = self.append_element(self.get_wave_propagation([distance] ,input_shape[1]), input_shape) 

        input_shape = self.append_element(self.get_wave_propagation([0.08], input_shape[1]),input_shape)
        #output
        input_shape = self.append_element(OL.LinearPolarizationFilter(angle = 0), input_shape)
        input_shape = self.append_element(OL.AbsFromJonesPolarizationLayer(), input_shape)
        input_shape = self.append_element(OL.AbsLayer(), input_shape)
        output_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)
        return
    