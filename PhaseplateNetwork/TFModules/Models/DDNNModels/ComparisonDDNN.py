import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN

class ComparisonDDNN(DDNN):
    def __init__(self,plates = 2, trainable_pixel = 120, plate_scale_factor = 1, propagation_size = 0.001792, propagation_pixel = 120, padding = 0.0015, frequency = 3.843e14, wavespeed = 3e8, **kwargs):
        super(ComparisonDDNN,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed, **kwargs)
        input_shape = [None, self.propagation_pixel, self.propagation_pixel, 1]
        #input handling
        #
        # if phase_input:
        #    input_shape = self.append_element(OL.IntensityToPhase(), input_shape)

        input_shape = self.append_element(self.get_padding_layer((input_shape[1],input_shape[2])), input_shape)
        #input_shape = self.append_element(OL.AmplitudeToIntensityJonesPolarization(axis), input_shape)
        #propagation and phaseplates
        input_shape = self.append_element(self.get_wave_propagation([0.08] ,input_shape[1]), input_shape)
        input_shape = self.append_element(OL.MultiplyWithSuperGaussian(input_shape[1:3]), input_shape)

        j = 0 
        for i in range(0,plates):
            #input_shape = self.append_element(self.get_wave_propagation(0.04, input_shape[1]), input_shape)
            #input_shape = self.append_element(OL.MultiplyWithSuperGaussian(input_shape[1:3]), input_shape)

            input_shape = self.append_element(self.get_phaseplate(False, True), input_shape)

            input_shape = self.append_element(self.get_wave_propagation(0.12, input_shape[1]),input_shape)
            input_shape = self.append_element(OL.MultiplyWithSuperGaussian(input_shape[1:3]), input_shape)


        #input_shape = self.append_element(self.get_wave_propagation(0.04, input_shape[1]), input_shape)
        #input_shape = self.append_element(OL.MultiplyWithSuperGaussian(input_shape[1:3]), input_shape)
        #input_shape = self.append_element(OL.LinearPolarizationFilter(angle=0.0), input_shape)
        #input_shape = self.append_element(OL.AbsFromJonesPolarizationLayer(),input_shape)
        input_shape = self.append_element(OL.AbsLayer(), input_shape)
        self.out_s = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)

        return
