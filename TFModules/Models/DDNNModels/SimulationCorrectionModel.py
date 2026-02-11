import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
import PhaseplateNetwork.TFModules.NetworkLayers as NL
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN
import PhaseplateNetwork.TFModules.OpticalLayers.Encodings as ENC
from PhaseplateNetwork.TFModules.Models.DDNNModels.DiffractionPlateNetwork import DiffractionPlateNetwork
from PhaseplateNetwork.TFModules.Models.DDNNModels.JonesNetworks.RotatingPlateNetwork import RotatingPlateNetwork

class SimulationCorrectionModel(DDNN):
    def __init__(self, num_layers = 3, propagation_distance = 0.08, intensity_output = True, encoding_pixels = 112, mean_size = 112, trainable_pixel=224, plate_scale_factor=1,
             propagation_size=0.001792, propagation_pixel=224, padding=0.001792, last_propagation = 0.13, sigma = 0.5, frequency = 3.843e14, wavespeed = 3e8,
             **kwargs):
                    
            super(SimulationCorrectionModel,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed, **kwargs)
    
            self.mean_size = mean_size
            #input_shape = [None, self.propagation_pixel, self.propagation_pixel, 1]
            #input handling
            input_shape = [None, 1]
            input_shape = self.append_element(ENC.RegressionEncodingPhase(encoding_pixels, sigma), input_shape)
            input_shape = self.append_element(OL.PaddingLayerPixels(encoding_pixels, propagation_pixel), input_shape)

            #print(input_shape)
            #if offset:
            #    input_shape = self.append_element(OL.LearnableAmplitudeAmplification(), input_shape)
            input_shape = self.append_element(self.get_padding_layer((input_shape[1],input_shape[2])), input_shape)
            #print(num_layers)

            #for i in range(0,num_layers):
                #p##rint(i)
            #    input_shape = self.append_element(self.get_wave_propagation([propagation_distance] ,input_shape[1]), input_shape)
            #    input_shape = self.append_element(self.get_phaseplate(False,True),input_shape)

            input_shape = self.append_element( self.get_wave_propagation([propagation_distance], input_shape[1]), input_shape)
            input_shape = self.append_element( self.get_wave_propagation([propagation_distance], input_shape[1]), input_shape)
            input_shape = self.append_element( self.get_wave_propagation([propagation_distance], input_shape[1]), input_shape) 

            self.nontrainable_phase_plate = self.get_phaseplate(False,False) 
            input_shape = self.append_element(self.nontrainable_phase_plate, input_shape)

            ##corretion phaseplate
            input_shape = self.append_element( self.get_wave_propagation([last_propagation/2.0], input_shape[1]), input_shape)
            self.correction_phaseplate = self.get_phaseplate(True,True)
            input_shape = self.append_element( self.correction_phaseplate, input_shape)
            input_shape = self.append_element(self.get_wave_propagation([last_propagation/2.0] ,input_shape[1]), input_shape)

            input_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)
            #input_shape = self.append_element(OL.AbsLayer(), input_shape)
            if intensity_output:
                input_shape = self.append_element(OL.IntensityLayer(), input_shape)
            else:
                input_shape = self.append_element(OL.AbsLayer(), input_shape)
            
            #output_shape = self.append_element(OL.MeanLayer((mean_size,mean_size)), input_shape)
            #self.scale_var = tf.Variable(1.0)
            #self.offset = offset

    def call(self,Input):
        x, phase = Input

        self.nontrainable_phase_plate.phases.assign(phase)
        u = tf.cast(x,dtype = tf.complex64)
        for layer in self.elements:
            u = layer.call(u)

        #u = self.average_pool(u)
        #u = u / self.scale_var
        return u
    
    def get_output_complex_field(self, Input):
        x, phase = Input

        self.nontrainable_phase_plate.phases.assign(phase)
        u = tf.cast(x,dtype = tf.complex64)
        for layer in self.elements[::-1]:
            u = layer.call(u)
        return u       
    
    
    def get_input_encoding(self, Input):
        x, phase = Input

        self.nontrainable_phase_plate.phases.assign(phase)
        u = tf.cast(x, dtype = tf.complex64)

        return self.elements[0](u)
