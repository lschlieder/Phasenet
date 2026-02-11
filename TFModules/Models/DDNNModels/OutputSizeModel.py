import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
import PhaseplateNetwork.TFModules.NetworkLayers as NL
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN
import PhaseplateNetwork.TFModules.OpticalLayers.Encodings as ENC
from PhaseplateNetwork.TFModules.Models.DDNNModels.DiffractionPlateNetwork import DiffractionPlateNetwork
from PhaseplateNetwork.TFModules.Models.DDNNModels.JonesNetworks.RotatingPlateNetwork import RotatingPlateNetwork

class OutputSizeModel(DDNN):
    def __init__(self, encoding = 'IntensityPhase', data_size= 28, num_coefficients = 4, categories = 10, radius = 0.0001, output_distance = 0.0002, num_layers = 3, output_function = 'sum', negative_output = False, output_shape = 'hex',  propagation_distance = 0.08, offset = False, intensity_output = False, trainable_pixel=112, plate_scale_factor=1,
             propagation_size=0.001792, propagation_pixel=112, padding=0.001792, last_propagation = 0.13, sigma = 0.5, frequency = 3.843e14, wavespeed = 3e8,
             **kwargs):
                    
            super(OutputSizeModel,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed, **kwargs)
    
            assert(encoding in ['IntensityPhase', 'Phase', 'Intensity', 'TrigPolynomial', 'TrigPolynomialPhase'])
            if encoding == 'IntensityPhase':
                self.input_size = propagation_pixel
                input_shape = [None, self.input_size, self.input_size, 1]
                input_shape = self.append_element(ENC.JointIntensityPhaseEncoding(), input_shape)
            elif encoding == 'Phase':
                self.input_size = propagation_pixel
                input_shape = [None, self.input_size, self.input_size, 1]
                input_shape = self.append_element(ENC.PhaseEncoding(), input_shape)
            elif encoding == 'Intensity':
                self.input_size = propagation_pixel
                input_shape = [None, self.input_size, self.input_size, 1]
                input_shape = self.append_element(ENC.IntensityEncoding(), input_shape)
            elif encoding =='TrigPolynomial':
                self.input_size = data_size
                input_shape = [None, self.input_size, self.input_size, 1]
                input_shape = self.append_element(ENC.TrigPolynomialEncoding(num_coefficients, (self.propagation_pixel, self.propagation_pixel)), input_shape)
            elif encoding =='TrigPolynomialPhase':
                self.input_size = data_size
                input_shape = [None, self.input_size, self.input_size, 1]
                input_shape = self.append_element(ENC.TrigPolynomialPhaseEncoding(num_coefficients, (self.propagation_pixel, self.propagation_pixel), image_size = propagation_size, sigma = sigma), input_shape)

            print('inp_shape: ', input_shape)
            input_shape = self.append_element(OL.PaddingLayerPixels(input_shape[1], propagation_pixel), input_shape)

            #print(input_shape)
            if offset:
                input_shape = self.append_element(OL.LearnableAmplitudeAmplification(), input_shape)
            input_shape = self.append_element(self.get_padding_layer((input_shape[1],input_shape[2])), input_shape)
            #print(num_layers)
            for i in range(0,num_layers):

                input_shape = self.append_element(self.get_wave_propagation([propagation_distance] ,input_shape[1]), input_shape)
                input_shape = self.append_element(self.get_phaseplate(False,True),input_shape)
            
            input_shape = self.append_element(self.get_wave_propagation([last_propagation] ,input_shape[1]), input_shape)

            input_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)
            #input_shape = self.append_element(OL.AbsLayer(), input_shape)
            if intensity_output:
                input_shape = self.append_element(OL.IntensityLayer(), input_shape)
            else:
                input_shape = self.append_element(OL.AbsLayer(), input_shape)
            output_shape = self.append_element(OL.CategoryOutputLayer(categories, (input_shape[1], input_shape[2]), [propagation_size + padding, propagation_size + padding], radius, output_distance, output_shape = output_shape, output_function = output_function, negative_output = negative_output),input_shape)
            
            

            self.offset = offset

    def call(self,Input):
        u = self.elements[0](Input)   


        for layer in self.elements[1:]:
            u = layer.call(u)
        return u
    
    def get_input_size(self):
        return [None, self.input_size, self.input_size, 1]
    

    @tf.function
    def get_output_image(self, Input):
        u = self.elements[0](Input)
        for layer in self.elements[1:-1]: 
            u = layer.call(u)
        return u
    
    def get_propagation_fields(self,Input):
        u = tf.cast(Input, dtype = tf.complex64)
        u_array = []
        u_array.append(u)
        for layer in self.elements[0:-1]:
            #print(layer.name)
            fields = layer.call(u)
            u = layer.call(u)
            u_array = u_array + [fields]
        return u_array
    
    def get_output_complex_field(self, Input):
        u = self.elements[0](Input)

        for layer in self.elements[1:-2]: 
            u = layer.call(u)
        return u       
    
    
    def get_input_encoding(self, Input):
        u = self.elements[0](Input)
        u = self.elements[1](u)
        return u
    
    def get_class_image(self, category):
        return self.elements[-1].get_wanted_output_image(category)
