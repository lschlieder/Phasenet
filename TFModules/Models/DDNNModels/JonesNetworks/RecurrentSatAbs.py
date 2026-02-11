import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN
#from PhaseplateNetwork.TFModules.OpticalLayers.LinearPolarizationFilter import LinearPolarizationFilter
class RecurrentSatAbs_v1(DDNN):
    def __init__(self, optical_density =1.0,Isat = 0.1,block_num = 4, trainable_polarization = 'xy', trainable_pixel = 112, plate_scale_factor = 2, propagation_size = 0.001792, propagation_pixel = 224, padding = 0.00175, frequency = 3.843e14, wavespeed = 3e8, **kwargs):
        super(RecurrentSatAbs_v1,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed, **kwargs)
        input_shape = [None, self.propagation_pixel, self.propagation_pixel, 2]
        #input handling
        #input_shape = self.append_element(self.get_padding_layer((input_shape[1],input_shape[2])), input_shape)
        #input_shape = self.append_element(OL.AmplitudeToIntensityJonesPolarization(axis), input_shape)
        #input_shape = self.append_element(self.get_optical_nonlinearity_sat_abs(0.001, input_shape[1]), input_shape)
        #input_shape = self.append_element(OL.SaturableAbsorber(depth = 0.00001, resolution = 0.00001, N = input_shape[1],L = propagation_size,
        #                                                      padding = 0.0,f0 = frequency, cM = wavespeed, OD = optical_density, I_saturation = Isat), input_shape)
        input_shape = self.append_element(OL.JonesSaturableAbsorber(optical_density,Isat, False), input_shape)
        input_shape = self.append_element(self.get_wave_propagation_without_padding([0.02], input_shape[1]), input_shape)
        input_shape = self.append_element(OL.LinearPolarizationFilter(angle = 0.0), input_shape) # Only let through x polarized light
        input_shape = self.append_element(OL.PolarizationRotator(angle=np.pi / 4), input_shape)
        #propagation and phaseplates
        for i in range(0,block_num):
            input_shape = self.append_element(self.get_wave_propagation_without_padding([0.04] ,input_shape[1]), input_shape)
            input_shape = self.append_element(self.get_jones_phaseplate(False,True, trainable_polarization),input_shape)

        output_shape = self.append_element(self.get_wave_propagation_without_padding([0.04] ,input_shape[1]), input_shape)

        #output_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)
        return
