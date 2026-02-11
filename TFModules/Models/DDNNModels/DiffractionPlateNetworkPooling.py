import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
import PhaseplateNetwork.TFModules.OpticalLayers.Encodings as ENC
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN

class DiffractionPlateNetworkPooling(DDNN):
    def __init__(self,pooling_factor = 4, num_layers = 10, phase_encoding = False, distance = 0.005, trainable_pixel = 120, plate_scale_factor = 1, propagation_size = 0.002, input_pixel = 30, padding = 0.002, frequency = 3.843e14, wavespeed = 3e8, **kwargs):
        super(DiffractionPlateNetworkPooling,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, pooling_factor*input_pixel, padding, frequency, wavespeed, **kwargs)
        self.input_pixel = input_pixel
        input_shape = [None, int(self.input_pixel), int(self.input_pixel), 1]
        
        #input handling
        #input_shape = self.append_element(tf.keras.layers.UpSampling2D(pooling_factor), input_shape)
        input_shape = self.append_element(OL.UpSampling2D(pooling_factor), input_shape)
        if phase_encoding:
            input_shape = self.append_element(ENC.PhaseEncoding(), input_shape)
        input_shape = self.append_element(self.get_padding_layer((input_shape[1],input_shape[2])), input_shape)
        print('added padding layer')
        #input_shape = self.append_element(OL.AmplitudeToIntensityJonesPolarization(axis), input_shape)
        #propagation and phaseplates
        for i in range(0,num_layers):
            input_shape = self.append_element(self.get_wave_propagation([distance] ,input_shape[1]), input_shape)
            input_shape = self.append_element(self.get_phaseplate(False,True),input_shape)
        #output
        #input_shape = self.append_element(OL.AbsFromJonesPolarizationLayer(), input_shape)
        input_shape = self.append_element(self.get_wave_propagation([distance], input_shape[1]), input_shape)
        input_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)
        output_shape = self.append_element(OL.PoolingLayer(factor = pooling_factor), input_shape)
        
        return
