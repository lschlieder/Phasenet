import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
import PhaseplateNetwork.TFModules.OpticalLayers.Encodings.PolarizationEncodings as PE
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN

class IntensityPolarizationFiltersResidualInput(DDNN):
    def __init__(self, axis = 'x', trainable_polarization = 'x',block_num = 4, distance = 0.04, trainable_pixel = 120, plate_scale_factor = 1, propagation_size = 0.001792, propagation_pixel = 120, padding = 0.001, frequency = 3.843e14, wavespeed = 3e8, **kwargs):
        super(IntensityPolarizationFiltersResidualInput,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed, **kwargs)
        input_shape = [None, self.propagation_pixel, self.propagation_pixel, 1]
        #input handling
        input_shape = self.append_element(self.get_padding_layer((input_shape[1],input_shape[2])), input_shape)
        input_shape = self.append_element(PE.LinearSingleDimensionEncoding(axis), input_shape)
        #propagation and phaseplates
    
        for i in range(0,block_num):
            for j in range(0,2):
                input_shape = self.append_element(self.get_wave_propagation([distance] ,input_shape[1]), input_shape)
                input_shape = self.append_element(self.get_jones_phaseplate(False,True, trainable_polarization),input_shape)
                input_shape = self.append_element(self.get_wave_propagation([distance] ,input_shape[1]), input_shape)
            input_shape = self.append_element(OL.LinearPolarizationFilter(angle = np.pi/4), input_shape)
            for j in range(0,2):
                input_shape = self.append_element(self.get_wave_propagation([distance] ,input_shape[1]), input_shape)
                input_shape = self.append_element(self.get_jones_phaseplate(False,True, trainable_polarization),input_shape)
                input_shape = self.append_element(self.get_wave_propagation([distance] ,input_shape[1]), input_shape) 

        input_shape = self.append_element(self.get_wave_propagation([0.08], input_shape[1]),input_shape)
        #output
        input_shape = self.append_element(OL.LinearPolarizationFilter(angle = 0), input_shape)
        input_shape = self.append_element(OL.AbsFromJonesPolarizationLayer(), input_shape)
        input_shape = self.append_element(OL.AbsLayer(), input_shape)
        output_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)

        self.amp_1 = tf.Variable(0.5, trainable=True)
        self.amp_2 = tf.Variable(0.5, trainable=True)
        return
    
    def call(self, inp):
        def add_ones(u, amplitude):
            amp = tf.abs(u[:,:,:,0:1])
            u_y = tf.complex(tf.ones_like(amp)*amplitude, 0.0) * tf.math.exp(1j * tf.complex(tf.zeros_like(amp), 0.0 ))
            return tf.concat((u[:,:,:,0:1], u_y), axis = 3)

        elem = self.elements
        u = tf.cast(inp,dtype = tf.complex64)

        u = elem[0].call(u)
        u = elem[1].call(u)
        j = 2
        for i in range(0,6):
            u = elem[j+i].call(u)
        
        u = add_ones(u,self.amp_1)
        u = elem[8].call(u)
        #print(u.shape)
        j = 9
        for i in range(0,12):
            u = elem[j+i].call(u)

        j = 9+12
        u = add_ones(u. self.amp_2)
        u = elem[21].call(u)
        j = 22

        for i in range(0,11):
            u = elem[j+i].call(u)
        #print(u.shape)

        return u
    
    def get_propagation_fields(self,inp):
        u = tf.cast(inp, dtype = tf.complex64)
        u_array = []
        u_array.append(u)
        def add_ones(u):
            amp = tf.abs(u[:,:,:,0:1])
            u_y = tf.complex(tf.ones_like(amp), 0.0) * tf.math.exp(1j * tf.complex(tf.zeros_like(amp), 0.0 ))
            return tf.concat((u[:,:,:,0:1], u_y), axis = 3)

        elem = self.elements
        u = tf.cast(inp,dtype = tf.complex64)

        u = elem[0].call(u)
        u_array = u_array + [u]
        u = elem[1].call(u)
        u_array = u_array + [u]
        j = 2
        for i in range(0,6):
            u = elem[j+i].call(u)
            u_array = u_array + [u]
        u = add_ones(u)

        u = elem[8].call(u)
        u_array = u_array + [u]
        #print(u.shape)
        j = 9
        for i in range(0,12):
            u = elem[j+i].call(u)
            u_array = u_array + [u]

        j = 9+12
        u = add_ones(u)
        u = elem[21].call(u)
        u_array = u_array + [u]
        j = 22

        for i in range(0,11):
            u = elem[j+i].call(u)
            u_array = u_array + [u]
        #print(u.shape)
            
        #u_array = u_array + [u]
        return u_array