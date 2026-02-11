import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
import PhaseplateNetwork.TFModules.NetworkLayers as NL
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN
import PhaseplateNetwork.TFModules.OpticalLayers.Encodings as ENC
from PhaseplateNetwork.TFModules.Models.DDNNModels.DiffractionPlateNetwork import DiffractionPlateNetwork
from PhaseplateNetwork.TFModules.Models.DDNNModels.JonesNetworks.RotatingPlateNetwork import RotatingPlateNetwork
import matplotlib.pyplot as plt 

class OpticalMultidimensionalRegression(DDNN):
    def __init__(self,output_dimension = 1, use_negative_weights = False, num_layers = 3, propagation_distance = [0.08], layers_trainable = [0,1,2], offset = False, intensity_output = False, encoding_pixels = 112, mean_size = 112, trainable_pixel=112, plate_scale_factor=1,
             propagation_size=0.001792, propagation_pixel=112, padding=0.001792, last_propagation = 0.13, sigma = 0.5, frequency = 3.843e14, wavespeed = 3e8,
             **kwargs):
                    
            super(OpticalMultidimensionalRegression,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed, **kwargs)
    
            self.num_layers = num_layers
            self.propagation_distance = propagation_distance
            self.layers_trainable = layers_trainable
            self.offset = offset
            self.intensity_output = intensity_output
            self.encoding_pixels = encoding_pixels
            self.mean_size = mean_size
            self.trainable_pixel = trainable_pixel
            self.plate_scale_factor = plate_scale_factor
            self.propagation_size = propagation_size
            self.propagaiton_pixel = propagation_pixel
            self.padding = padding
            self.last_propagation = last_propagation
            self.sigma = sigma
            self.frequency = frequency
            self.wavespeed = wavespeed
            #self.mean_size = mean_size
            #input_shape = [None, self.propagation_pixel, self.propagation_pixel, 1]
            #input handling
            input_shape = [None, 1]
            input_shape = self.append_element(ENC.RegressionEncodingBoth(encoding_pixels, sigma), input_shape)
            input_shape = self.append_element(OL.PaddingLayerPixels(encoding_pixels, propagation_pixel), input_shape)

            if len(propagation_distance) == 1:
                propagation_distance = np.ones(num_layers) * propagation_distance

            #print(input_shape)
            if offset:
                input_shape = self.append_element(OL.LearnableAmplitudeAmplification(), input_shape)
            input_shape = self.append_element(self.get_padding_layer((input_shape[1],input_shape[2])), input_shape)
            #print(num_layers)
            self.plates = []
            for i in range(0,num_layers):
                input_shape = self.append_element(self.get_wave_propagation([propagation_distance[i]] ,input_shape[1]), input_shape)
                if i in layers_trainable:
                    plate = self.get_phaseplate(False, True)
                    self.plates.append(plate)
                    input_shape = self.append_element(plate, input_shape)

                    #input_shape = self.append_element(self.get_wave_propagation([propagation_distance] ,input_shape[1]), input_shape)
                    #input_shape = self.append_element(self.get_phaseplate(False,True),input_shape)
                else:
                    plate = self.get_phaseplate(False, False)
                    self.plates.append(plate)
                    input_shape = self.append_element(plate,input_shape)
            
            input_shape = self.append_element(self.get_wave_propagation([last_propagation] ,input_shape[1]), input_shape)

            input_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)
            #input_shape = self.append_element(OL.AbsLayer(), input_shape)
            if intensity_output:
                input_shape = self.append_element(OL.IntensityLayer(), input_shape)
            else:
                input_shape = self.append_element(OL.AbsLayer(), input_shape)
            #output_shape = self.append_element(OL.MeanLayer((mean_size,mean_size)), input_shape)
            if use_negative_weights == True:
                output_dimension = output_dimension * 2

            output_shape = self.append_element(OL.CategoryOutputLayer(output_dimension, (input_shape[1], input_shape[2]), [propagation_size + padding, propagation_size + padding], radius, distance, output_shape = output_shape, output_function = output_function, negative_output = negative_output),input_shape)
           
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
        for layer in self.elements[:-1]: 
            u = layer.call(u)
        return u
    
    def get_output_complex_field(self, Input):
        if self.offset:
            u = tf.cast(Input,dtype = tf.complex64) * tf.complex(self.scale_var,0.0)
        else:
            u = tf.cast(Input,dtype = tf.complex64)
        for layer in self.elements[:-2]: 
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
        for c in self.plates:
            phases.append(c.phases)

        phases_arr = np.squeeze(np.array(phases))
        #assert( phases_arr.shape[0] == self.num_layers)
        assert( len(phases_arr.shape) == 3)
        np.save(path, phases_arr)

    def get_phaseplates(self):
        plates = []
        for c in self.plates:
            plates.append(c.phases)
        
        plates = np.squeeze(np.array(plates))
        return plates

    def plot_phaseplates(self, fig = None, ax = None, **kwargs):
        num = len(self.plates)
        fig, ax = plt.subplots( 1, num, figsize = (num*5, 5))
        for i,p in enumerate(self.plates):
            img = ax[i].imshow(p.phases[0,:,:,0],**kwargs)
            fig.colorbar(img, ax = ax[i])

        return fig, ax 
    

    def get_field_at_position(self, Input, pos):
        if self.offset:
            u = tf.cast(Input,dtype = tf.complex64) * tf.complex(self.scale_var,0.0)
        else:
            u = tf.cast(Input,dtype = tf.complex64)
        

        z = 0.0
        i = 0 
        while z < pos:
            layer = self.elements[i]
            if hasattr(layer, 'z'):
                z = z + layer.z[0]
            
            if z < pos:
                u = layer.call(u)
            else:
                prop_size = pos - (z - layer.z[0])
                last_prop_layer = self.get_wave_propagation([prop_size] ,self.output_shapes[i-1][1])
                u = last_prop_layer.call(u)

            i = i+1


        #u = self.average_pool(u)
        #u = u / self.scale_var
        return u
    
    
    def get_propagation_distance(self):
        z = 0.0
        for layer in self.elements:
            if hasattr(layer, 'z'):
                z = z + layer.z[0]
        return z
    
    def get_plate_positions(self):
        z = 0.0
        dist_arr = []
        for layer in self.elements:
            if hasattr(layer, 'z'):
                z = z + layer.z[0]
            if isinstance(layer, OL.PhasePlate):
                dist_arr.append(z)
        return dist_arr
            


    def get_config(self):
        temp = {
            "num_layers": self.num_layers,
            "propagation_distance" : self.propagation_distance,
            "layers_trainable": self.layers_trainable,
            "offset" : self.offset,
            "intensity_output" : self.intensity_output,
            "encoding_pixels" : self.encoding_pixels,
            "mean_size" : self.mean_size,
            "trainable_pixel": self.trainable_pixel,
            "plate_scale_factor": self.plate_scale_factor,
            "propagation_size" : self.propagation_size,
            "propagation_pixel" : self.propagation_pixel,
            "padding" : self.padding,
            "last_propagation" : self.last_propagation,
            "sigma" : self.sigma,
            "frequency": self.frequency,
            "wavespeed": self.wavespeed

        }
        return temp

    @classmethod
    def from_config(cls, config):
        return cls(**config)
