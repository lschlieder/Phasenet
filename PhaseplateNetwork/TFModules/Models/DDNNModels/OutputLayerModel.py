import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
import PhaseplateNetwork.TFModules.NetworkLayers as NL
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN
import PhaseplateNetwork.TFModules.OpticalLayers.Encodings as ENC
from PhaseplateNetwork.TFModules.Models.DDNNModels.DiffractionPlateNetwork import DiffractionPlateNetwork
from PhaseplateNetwork.TFModules.Models.DDNNModels.JonesNetworks.RotatingPlateNetwork import RotatingPlateNetwork
from PhaseplateNetwork.TFModules.Models.DDNNModels.OutputSizeModel import OutputSizeModel
import matplotlib.pyplot as plt

class OutputLayerModel(DDNN):
    def __init__(self, num_hyper_layers = 3, encoding = 'IntensityPhase', data_size= 28, categories = [100, 100,10] , radius = [0.0001, 0.0001, 0.0002], output_distance = [0.0002, 0.0002,0.0006], num_layers = 3,
                  output_function = 'sum', negative_output = False, output_shape = 'hex',  propagation_distance = 0.08, offset = False, intensity_output = False,
                    trainable_pixel=112, plate_scale_factor=1, propagation_size=0.001792, propagation_pixel=112, padding=0.001792, last_propagation = 0.13,
                      sigma = 0.5, frequency = 3.843e14, wavespeed = 3e8, **kwargs):
                    
            super(OutputLayerModel,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed, **kwargs)
            if encoding == 'IntensityPhase':
                self.input_size = propagation_pixel
            elif encoding == 'Phase':
                self.input_size = propagation_pixel
            elif encoding == 'Intensity':
                self.input_size = propagation_pixel
            elif encoding =='TrigPolynomial':
                self.input_size = data_size
            
            input_shape = [None, self.input_size, self.input_size, 1]



            if isinstance(num_layers, float) or isinstance(num_layers, int):
                num_layers = np.repeat(num_layers, num_hyper_layers)

            print(categories)
            assert(len(categories) == num_hyper_layers)
            assert(len(radius) == num_hyper_layers)
            assert(len(output_distance) == num_hyper_layers)
            assert(len(num_layers) == num_hyper_layers)

            models = []
            
            for n in range(0,num_hyper_layers):
                model = OutputSizeModel(encoding, data_size, categories[n], radius[n], output_distance[n], num_layers[n], output_function, negative_output, output_shape, 
                                        propagation_distance, offset, intensity_output, trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding,
                                        last_propagation, sigma, frequency, wavespeed, **kwargs)  

                output_enc = model.elements[-1].get_joint_masks()
                plt.figure()
                plt.imshow(output_enc)
                plt.show()     
                self.append_element(model, input_shape)
                if n < (num_hyper_layers-1):
                    self.append_element(ENC.FlatToImage((input_shape[1], input_shape[2])), input_shape)


    def call(self, input):
        u = input
        for e in self.elements:
            u = e.call(u)
        return u
            
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
            #fields = layer.call(u)
            fields = layer.get_propagation_fields(u)
            #if len(fields.shape()) == 4:
            #print(fields)
            u_array = u_array + fields
            u = layer.call(u)
        return u_array
            
    def get_class_image(self, category):
        return self.elements[-1].get_class_image(category)
            

