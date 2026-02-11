import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN
#from PhaseplateNetwork.TFModules.OpticalLayers.LinearPolarizationFilter import LinearPolarizationFilter
class TestDepthNetwork(DDNN):
    def __init__(self, plates = 8,distance = 0.04, trainable_pixel=112, plate_scale_factor=2, propagation_size=0.001792,
                 propagation_pixel=224, padding=0.00175, frequency=3.843e14, wavespeed=3e8, **kwargs):
        super(TestDepthNetwork,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed, **kwargs)

        input_shape = [None, self.propagation_pixel, self.propagation_pixel, 1]
        # input handling
        input_shape = self.append_element(OL.AmplitudeToPhase, input_shape)
        input_shape = self.append_element(self.get_padding_layer((input_shape[1], input_shape[2])), input_shape)
        d_plate = distance / plates
        for i in range(0,plates):
            input_shape = self.append_element(self.get_wave_propagation(d_plate, input_shape[1]), input_shape)
            input_shape = self.append_element(self.get_phaseplate(False, True), input_shape)
        input_shape= self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)

        return