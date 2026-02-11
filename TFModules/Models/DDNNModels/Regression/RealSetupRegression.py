import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
import PhaseplateNetwork.TFModules.NetworkLayers as NL
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN
import PhaseplateNetwork.TFModules.OpticalLayers.Encodings as ENC
from PhaseplateNetwork.TFModules.Models.DDNNModels.DiffractionPlateNetwork import DiffractionPlateNetwork
from PhaseplateNetwork.TFModules.Models.DDNNModels.JonesNetworks.RotatingPlateNetwork import RotatingPlateNetwork

class RealSetupRegression(DDNN):
    def __init__(self, image_input = False,  num_layers = 3, layers_trainable = [0,1,2], amp = True, phase = True, use_realistic_plates = False, propagation_distance = [0.08, 0.08, 0.08], offset = False, intensity_output = True, encoding_pixels = 112, mean_size = 112, trainable_pixel=112, plate_scale_factor=1,
             propagation_size=0.001792, propagation_pixel=112, padding=0.001792, last_propagation = 0.13, sigma = 0.5, frequency = 3.843e14, wavespeed = 3e8,
             **kwargs):
                    
            super(RealSetupRegression,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed, **kwargs)
    
            self.mean_size = mean_size
            self.num_layers = num_layers
            #input_shape = [None, self.propagation_pixel, self.propagation_pixel, 1]
            #input handling
            input_shape = [None, encoding_pixels, encoding_pixels, 1]
            if image_input == False:
                input_shape = [None, 1]
                input_shape = self.append_element(ENC.RegressionEncodingBoth(encoding_pixels, sigma), input_shape)
                
            input_shape = self.append_element(OL.PaddingLayerPixels(encoding_pixels, propagation_pixel), input_shape)

            #print(input_shape)
            if offset:
                input_shape = self.append_element(OL.LearnableAmplitudeAmplification(), input_shape)
            input_shape = self.append_element(self.get_padding_layer((input_shape[1],input_shape[2])), input_shape)
            #print(num_layers)
            self.correction_plates = []
            for i in range(0,num_layers):
                #p##rint(i)
                input_shape = self.append_element(self.get_wave_propagation([propagation_distance[i]] ,input_shape[1]), input_shape)

                if i in layers_trainable:
                    corr_plate = OL.PhasePlate( [propagation_pixel,propagation_pixel], 1, amp, phase)

                    self.correction_plates.append(corr_plate)
                    #input_shape = self.append_element(self.get_phaseplate(False,True),input_shape)
                    input_shape = self.append_element(corr_plate, input_shape)
                else:
                    #if using_reflection_plates: 
                        #corr_plate = OL.PhasePlateWithReflection(shape = )

                    #else:
                    corr_plate = OL.PhasePlate( [propagation_pixel,propagation_pixel], 1, False, False)
                    self.correction_plates.append(corr_plate)
                    #input_shape = self.append_element(self.get_phaseplate(False,True),input_shape)
                    input_shape = self.append_element(corr_plate, input_shape)
                    #input_shape = self.append_element(self.get_phaseplate(False,False), input_shape)
            
            input_shape = self.append_element(self.get_wave_propagation([last_propagation] ,input_shape[1]), input_shape)

            input_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)
            print(propagation_pixel)
            print(encoding_pixels)
            input_shape = self.append_element(OL.CroppingLayerPixels(propagation_pixel, encoding_pixels), input_shape)
            #input_shape = self.append_element(OL.AbsLayer(), input_shape)
            if intensity_output:
                input_shape = self.append_element(OL.IntensityLayer(), input_shape)
            else:
                input_shape = self.append_element(OL.AbsLayer(), input_shape)

            #input_shape = self.append_element(OL.Crop)
            #output_shape = self.append_element(OL.MeanLayer((mean_size,mean_size)), input_shape)
            self.scale_var = tf.Variable(1.0)
            self.offset = offset

    def call(self,Input):
        if self.offset:
            u = tf.cast(Input,dtype = tf.complex64) * tf.complex(self.scale_var,0.0)
        else:
            u = tf.cast(Input,dtype = tf.complex64)
        for layer in self.elements:
            u = layer.call(u)
        #u = self.average_pool(u)
        #u = u / self.scale_var
        return u
    
    def get_output_image(self, Input):
        if self.offset:
            u = tf.cast(Input,dtype = tf.complex64) * tf.complex(self.scale_var,0.0)
        else:
            u = tf.cast(Input,dtype = tf.complex64)
        for layer in self.elements: 
            u = layer.call(u)
        return u
    
    def get_output_complex_field(self, Input):
        if self.offset:
            u = tf.cast(Input,dtype = tf.complex64) * tf.complex(self.scale_var,0.0)
        else:
            u = tf.cast(Input,dtype = tf.complex64)
        for layer in self.elements[:-1]: 
            u = layer.call(u)
        return u       
    
    
    def get_input_encoding(self, Input):

        if self.offset:
            u = tf.cast(Input,dtype = tf.complex64) * tf.complex(self.scale_var,0.0)
        else:
            u = tf.cast(Input,dtype = tf.complex64)

        return self.elements[0](u)
    

    def save_phaseplate(self,path = "./phaseplates_real_hologram_setup.npy"):
        phases = []
        for c in self.correction_plates:
            phases.append(c.phases)

        phases_arr = np.squeeze(np.array(phases))
        assert( phases_arr.shape[0] == self.num_layers)
        assert( len(phases_arr.shape) == 3)
        np.save(path, phases_arr)
