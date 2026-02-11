import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN

class NonlinearAcousticProjector(DDNN):
    def __init__(self,bias = 0.1, num_plates = 2, phase_input = False,starting_distance = 0.03, distance = 0.03, trainable_pixel = 60, plate_scale_factor = 3, propagation_size = 0.05, propagation_pixel = 180, padding = 0.05, frequency = 1.00e6, wavespeed = 1484, **kwargs):
        super(NonlinearAcousticProjector,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed, **kwargs)
        input_shape = [None, self.propagation_pixel, self.propagation_pixel, 1]
        #input handling
        if phase_input:
            input_shape = self.append_element(OL.IntensityToPhase(), input_shape)
        input_shape = self.append_element(self.get_padding_layer((input_shape[1],input_shape[2])), input_shape)
        #input_shape = self.append_element(OL.AmplitudeToIntensityJonesPolarization(axis), input_shape)
        #propagation and phaseplates
        input_shape = self.append_element(self.get_wave_propagation([starting_distance], input_shape[1]), input_shape)
        for i in range(0,num_plates):
            input_shape = self.append_element(OL.ReluLayer(bias), input_shape)
            input_shape = self.append_element(self.get_phaseplate(False,True),input_shape)
            input_shape = self.append_element(self.get_wave_propagation([distance], input_shape[1]),input_shape)
        outputput_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)
        return
