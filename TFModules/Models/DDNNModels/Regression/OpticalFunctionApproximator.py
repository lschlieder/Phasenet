import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
import PhaseplateNetwork.TFModules.NetworkLayers as NL
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN
import PhaseplateNetwork.TFModules.OpticalLayers.Encodings.PolarizationEncodings as PE
from PhaseplateNetwork.TFModules.Models.DDNNModels.DiffractionPlateNetwork import DiffractionPlateNetwork
from PhaseplateNetwork.TFModules.Models.DDNNModels.JonesNetworks.RotatingPlateNetwork import RotatingPlateNetwork

class OpticalFunctionApproximator(DDNN):
    def __init__(self, num_layers = 4, propagation_distance = 0.3,mean_size = 112, trainable_pixel=112, plate_scale_factor=1,
             propagation_size=0.001792, propagation_pixel=112, padding=0.001792, frequency=3.843e14, wavespeed = 3e8,
             **kwargs):
                    
            super(OpticalFunctionApproximator,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed, **kwargs)
    
            
            #input_shape = [None, self.propagation_pixel, self.propagation_pixel, 1]
            #input handling
            input_shape = [None, 1]
            input_shape = self.append_element(PE.RegressionEncoding(self.propagation_pixel), input_shape)
            #print(input_shape)
            input_shape = self.append_element(self.get_padding_layer((input_shape[1],input_shape[2])), input_shape)
            #input_shape = self.append_element(OL.AmplitudeToJonesPolarizationLayer(), input_shape)
            #propagation and phaseplates
            for i in range(0,num_layers):
                input_shape = self.append_element(self.get_wave_propagation([propagation_distance] ,input_shape[1]), input_shape)
                input_shape = self.append_element(self.get_jones_phaseplate(False,True, 'xy'),input_shape)
            #output
            input_shape = self.append_element(self.get_wave_propagation([propagation_distance] ,input_shape[1]), input_shape)
            input_shape = self.append_element(OL.LinearPolarizationFilter(angle = np.pi/4), input_shape)

            input_shape = self.append_element(OL.LinearPolarizationFilter( angle = 0.0), input_shape)

            input_shape = self.append_element(OL.AbsFromJonesPolarizationLayer(), input_shape)
            input_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)
            input_shape = self.append_element(OL.AbsLayer(), input_shape)
            output_shape = self.append_element(OL.MeanLayer((mean_size, mean_size), input_shape)

            self.scale_var = tf.Variable(1.0)

    def call(self,Input):
        u = tf.cast(Input,dtype = tf.complex64)
        #u = repeat_image_tensor(u,self.scale)
        for layer in self.elements:
            #print(u.shape)
            #print(u.dtype)
            u = layer.call(u)
        #u = self.average_pool(u)
        u = u / self.scale_var
        return u


                



