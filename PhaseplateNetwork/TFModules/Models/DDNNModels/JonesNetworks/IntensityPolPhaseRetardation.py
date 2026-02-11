import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN

class IntensityPolPhaseRetardation(DDNN):
    def __init__(self, axis = 'x', trainable_polarization = 'x', trainable_pixel = 112, plate_scale_factor = 2, propagation_size = 0.001792, propagation_pixel = 224, padding = 0.00175, frequency = 3.843e14, wavespeed = 3e8, **kwargs):
        super(IntensityPolPhaseRetardation,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed, **kwargs)
        input_shape = [None, self.propagation_pixel, self.propagation_pixel, 1]
        #input handling
        input_shape = self.append_element(self.get_padding_layer((input_shape[1],input_shape[2])), input_shape)
        input_shape = self.append_element(OL.AmplitudeToIntensityJonesPolarization(axis), input_shape)
        #propagation and phaseplates
        input_shape = self.append_element(self.get_wave_propagation([0.04] ,input_shape[1]), input_shape)
        input_shape = self.append_element(self.get_jones_phaseplate(False,True, trainable_polarization),input_shape)
        input_shape = self.append_element(OL.PhaseRetarder(),input_shape)
        input_shape = self.append_element(self.get_wave_propagation([0.08], input_shape[1]),input_shape)
        input_shape = self.append_element(self.get_jones_phaseplate(False,True, trainable_polarization), input_shape)
        input_shape = self.append_element(OL.PhaseRetarder(),input_shape)
        input_shape = self.append_element(self.get_wave_propagation([0.08], input_shape[1]),input_shape)
        input_shape = self.append_element(OL.LinearPolarizationFilter(angle = np.pi/4), input_shape)
        input_shape = self.append_element(self.get_jones_phaseplate(False,True, trainable_polarization), input_shape)
        input_shape = self.append_element(self.get_wave_propagation([0.08], input_shape[1]),input_shape)
        input_shape = self.append_element(self.get_jones_phaseplate(False,True, trainable_polarization), input_shape)
        input_shape = self.append_element(OL.PhaseRetarder(),input_shape)
        input_shape = self.append_element(self.get_wave_propagation([0.13], input_shape[1]),input_shape)
        #output
        input_shape = self.append_element(OL.CircularPolarizationFilter(),input_shape)
        input_shape = self.append_element(OL.AbsFromJonesPolarizationLayer(), input_shape)
        output_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)
        return
