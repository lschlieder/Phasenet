import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
import PhaseplateNetwork.TFModules.OpticalLayers.Encodings.PolarizationEncodings as ENC
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN


# from PhaseplateNetwork.TFModules.OpticalLayers.LinearPolarizationFilter import LinearPolarizationFilter
class RotatingPlateNetwork(DDNN):
    def __init__(self,encoding = 'linear_intensity_xy',num_layers = 4, phase_input = False, distance = 0.04, trainable_pixel = 112, plate_scale_factor = 2, propagation_size = 0.001792, propagation_pixel = 224, padding = 0.00175, frequency = 3.843e14, wavespeed = 3e8, **kwargs):
        super(RotatingPlateNetwork, self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel,
                                               padding, frequency, wavespeed, **kwargs)
        input_shape = [None, self.propagation_pixel, self.propagation_pixel, 1]
        input_shape = self.append_element( ENC.PolarizationEncoding(encoding), input_shape)

        #input_shape = self.append_element(self.get_wave_propagation(0.04, input_shape[1]),input_shape)
        #input_shape = self.append_element(OL.MultiplyWithSuperGaussian(input_shape[1:3]), input_shape)
        for i in range(0,num_layers):
            #input_shape = self.append_element(self.get_jones_phaseplate(False, True, 'x'),input_shape)
            input_shape = self.append_element(self.get_wave_propagation(distance, input_shape[1]), input_shape)
            #input_shape = self.append_element(OL.MultiplyWithSuperGaussian(input_shape[1:3]), input_shape)

            input_shape = self.append_element(self.get_rotating_plate(), input_shape)
            

            ##input_shae = self.append_element(self.get_jones_phaseplate(False, True,'xy'), input_shape)
            #input_shape = self.append_element(self.get_wave_propagation(0.08, input_shape[1]),input_shape)
            #input_shape = self.append_element(OL.MultiplyWithSuperGaussian(input_shape[1:3]), input_shape)


        #input_shape = self.append_element(self.get_rotating_plate(), input_shape)
        input_shape = self.append_element(self.get_wave_propagation(distance, input_shape[1]), input_shape)
        #input_shape = self.append_element(OL.MultiplyWithSuperGaussian(input_shape[1:3]), input_shape)
        input_shape = self.append_element(OL.LinearPolarizationFilter(angle=0.0), input_shape)
        input_shape = self.append_element(OL.AbsFromJonesPolarizationLayer(),input_shape)
        output_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)

        return
