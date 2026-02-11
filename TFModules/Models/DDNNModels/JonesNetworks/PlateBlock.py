import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN
#from PhaseplateNetwork.TFModules.OpticalLayers.LinearPolarizationFilter import LinearPolarizationFilter
class PlateBlock(DDNN):
    def __init__(self, trainable_polarization = 'xy', block_num = 6, distance = 0.04, trainable_pixel = 112, plate_scale_factor = 2, propagation_size = 0.001792, propagation_pixel = 224, padding = 0.002, frequency = 3.843e14, wavespeed = 3e8, **kwargs):
        super(PlateBlock,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed, **kwargs)
        input_shape = [None, self.propagation_pixel, self.propagation_pixel, 2]
        #input_shape = self.append_element(self.get_padding_layer((input_shape[1],input_shape[2])), input_shape)
        for i in range(1, block_num+1):
            input_shape = self.append_element(self.get_wave_propagation_without_padding([distance], input_shape[1]),
                                              input_shape)
            #if not i %3 == 0:
            #    input_shape = self.append_element(OL.PhaseRetarder(np.pi/3), input_shape)
            #else:
            #    input_shape = self.append_element(OL.LinearPolarizationFilter(angle=np.pi / 4), input_shape)
            input_shape = self.append_element(self.get_jones_phaseplate(False, True, trainable_polarization),
                                              input_shape)
        output_shape = self.append_element(self.get_wave_propagation_without_padding([distance], input_shape[1]),
                                              input_shape)
        #output_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)

        return

    def get_input_shape(self):
        return [None, self.propagation_pixel, self.propagation_pixel, 2]
