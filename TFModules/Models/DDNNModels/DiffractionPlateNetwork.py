import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
import PhaseplateNetwork.TFModules.OpticalLayers.Encodings as EN
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN


class DiffractionPlateNetwork(DDNN):
    def __init__(self,num_layers = 4, phase_input = False, distance = 0.04, trainable_pixel = 112, plate_scale_factor = 2, 
                 propagation_size = 0.001792, propagation_pixel = 224, padding = 0.00175, frequency = 3.843e14,
                 wavespeed = 3e8, init = 'uniform', **kwargs):
        super(DiffractionPlateNetwork,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed, **kwargs)
        input_shape = [None, self.propagation_pixel, self.propagation_pixel, 1]
        
        #input handling
        if phase_input:
            #input_shape = self.append_element(OL.IntensityToPhase(), input_shape)
            input_shape = self.append_element(EN.PhaseEncoding(), input_shape)
        input_shape = self.append_element(self.get_padding_layer((input_shape[1],input_shape[2])), input_shape)
        #input_shape = self.append_element(OL.AmplitudeToIntensityJonesPolarization(axis), input_shape)
        #propagation and phaseplates
        self.num_layers = num_layers
        for i in range(0,num_layers):
            input_shape = self.append_element(self.get_wave_propagation([distance] ,input_shape[1]), input_shape)
            plate = OL.PhasePlate([trainable_pixel,trainable_pixel], plate_scale_factor, False, True, init)
 
            #input_shape = self.append_element(self.get_phaseplate(False,True),input_shape)
            input_shape = self.append_element(plate, input_shape)

        #output
        #input_shape = self.append_element(OL.AbsFromJonesPolarizationLayer(), input_shape)
        input_shape = self.append_element(self.get_wave_propagation([distance], input_shape[1]), input_shape)
        
        input_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)
        output_shape = self.append_element(OL.IntensityLayer(), input_shape)
        return