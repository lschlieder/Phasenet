import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN
import PhaseplateNetwork.TFModules.OpticalLayers.Encodings.PolarizationEncodings as PE
from PhaseplateNetwork.TFModules.Models.DDNNModels.DiffractionPlateNetwork import DiffractionPlateNetwork
from PhaseplateNetwork.TFModules.Models.DDNNModels.JonesNetworks.RotatingPlateNetwork import RotatingPlateNetwork

class OpticalRegression(DDNN):
        def __init__(self, num_layers = 4, propagation_distance = 0.3, phase_input = False, trainable_pixel=800, plate_scale_factor=1,
                 propagation_size=0.0144, propagation_pixel=800, padding=0.0144, frequency=5.639097744e14, wavespeed=3e8,
                 **kwargs):
                        
                super(OpticalRegression,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed, **kwargs)
        
                
                #self.optical_network = DiffractionPlateNetwork( num_layers, phase_input, propagation_distance, trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel,
                #                                                padding, frequency, wavespeed  )
                self.optical_network = RotatingPlateNetwork( 'linear_intensity_xy', num_layers, phase_input, propagation_distance, trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel,
                                                                padding, frequency, wavespeed  )
                self.propagation_pixel = propagation_pixel

                

        def call(self, input):

                #print(input.shape)
                inp_image = tf.reshape(input, (-1, 1,1,1))
                inp_image = tf.tile( inp_image, ( 1, self.propagation_pixel, self.propagation_pixel, 1))
                #inp_image = tf.image.pad_to_bounding_box(inp_image, int(self.propagation_pixel/2.0), int(self.propagation_pixel/2.0), self.propagation_pixel, self.propagation_pixel )
                
                
                
                optical_out = tf.math.abs(self.optical_network(inp_image))

                window_low = int(self.propagation_pixel/2.0 - self.propagation_pixel/10.0)
                window_high = int(self.propagation_pixel/2.0 + self.propagation_pixel/10.0)
                single = int(self.propagation_pixel/2.0)
                #out_num = tf.reduce_mean( optical_out[:,single:single+1, single:single+1,:  ], axis = [1,2,3])
                out_num = tf.reduce_mean( optical_out[:,window_low:window_high,window_low:window_high ,:  ], axis = [1,2,3])
                return out_num 
        
        def get_image_variables(self):
                return self.optical_network.get_image_variables()
        
        def get_propagation_fields(self,input):
                inp_image = tf.reshape(input, (-1, 1,1,1))
                inp_image = tf.tile( inp_image, ( 1, self.propagation_pixel, self.propagation_pixel, 1))
                return self.optical_network.get_propagation_fields(inp_image)

