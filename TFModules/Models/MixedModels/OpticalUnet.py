import tensorflow as tf
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN
import PhaseplateNetwork.TFModules.OpticalLayers as OL
import PhaseplateNetwork.TFModules.OpticalLayers.Encodings as EN
import numpy as np


class DiffractionPlateNetwork(DDNN):
    def __init__(self,num_layers = 4, mu = 0.000008, size = (112,112), phase_input = True, distance = 0.08 ,
                frequency = 3.843e14,
                 wavespeed = 3e8, **kwargs):
        
        self.size = size

        plate_scale_factor = 1
        trainable_pixel = size
        propagation_pixel = size


        propagation_size = np.array(size) * mu
        print(propagation_size)
        padding = propagation_size
        padding = 0.0
        super(DiffractionPlateNetwork,self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed, **kwargs)
        input_shape = [None, self.propagation_pixel, self.propagation_pixel, 1]
        
        #input handling

        if phase_input:
            #input_shape = self.append_element(OL.IntensityToPhase(), input_shape)
            input_shape = self.append_element(EN.PhaseEncoding(), input_shape)
        input_shape = self.append_element(self.get_padding_layer((input_shape[1],input_shape[2])), input_shape)
        #input_shape = self.append_element(OL.AmplitudeToIntensityJonesPolarization(axis), input_shape)
        #propagation and phaseplates
        for i in range(0,num_layers):
            input_shape = self.append_element(self.get_wave_propagation([distance] ,input_shape[1]), input_shape)
            input_shape = self.append_element(self.get_phaseplate(False,True),input_shape)
        #output
        #input_shape = self.append_element(OL.AbsFromJonesPolarizationLayer(), input_shape)
        input_shape = self.append_element(self.get_wave_propagation([distance/2.0], input_shape[1]), input_shape)
        
        input_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)
        output_shape = self.append_element(OL.IntensityLayer(), input_shape)
        return

class conv_layer(tf.keras.layers.Layer):
    def __init__(self,out_blocks = 8, middle_blocks = 4, activation = 'relu', **kwargs):
        super(conv_layer,self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(middle_blocks, (5,5), activation = 'relu', padding = 'same')
        self.conv2 = tf.keras.layers.Conv2D(out_blocks, (5,5), padding = 'same', activation = activation)
        
        
    def call(self, input):
        return self.conv2(self.conv1(input))
    
class up_conv_layer(tf.keras.layers.Layer):
    def __init__(self,out_blocks = 8, **kwargs):
        super(up_conv_layer,self).__init__(**kwargs)
        
        self.conv1 = tf.keras.layers.Conv2DTranspose(out_blocks*2, (5,5), activation = 'relu', padding = 'same')
        self.conv2 = tf.keras.layers.Conv2DTranspose(out_blocks, (5,5),strides = (2,2),  activation = 'relu',padding = 'same')
        
    def call(self,input):
        return self.conv2(self.conv1(input))
    

class DDNN_block(tf.keras.layers.Layer):
    def __init__(self, num_blocks = 8, size = (112,112), **kwargs):
        super(DDNN_block, self).__init__(**kwargs)

        networks = []
        for i in range(0,num_blocks):
            networks.append ( DiffractionPlateNetwork(size = size))
        self.networks = networks

    def call(self, input, **kwargs):
        u = input
        out = []
        for n in self.networks:
            o_n = n(u)
            out.append(o_n)

        return tf.concat(out, axis = 3)
    

class unet(tf.keras.models.Model):
    def __init__(self, mu = 0.000008, **kwargs):
        super(unet, self).__init__(**kwargs)
        

        self.layer1 = conv_layer(out_blocks = 32, middle_blocks = 1)


        self.layer1 = DiffractionPlateNetwork(num_layers = 3, size = (28,28))
        self.down_1 = tf.keras.layers.MaxPool2D(padding = 'same')
        
        self.layer2 = conv_layer(out_blocks = 64, middle_blocks = 2)
        self.down_2 = tf.keras.layers.MaxPool2D(padding = 'same')
        
        #self.layer3 = conv_layer(out_blocks = 8, middle_blocks = 4)
        #self.down_3 = tf.keras.layers.MaxPool2D(padding = 'same')
        
        
        #self.up_3 = up_conv_layer(out_blocks=4)
        self.up_2 = up_conv_layer(out_blocks=32)
        self.up_1 = up_conv_layer(out_blocks=16)
        
        
        self.t_encoding = tf.keras.layers.Dense(28*28)
        
        self.last_conv = conv_layer(out_blocks = 1, middle_blocks = 10, activation = 'linear')
        
    def call(self, input):
        img, t = input
        
        t_enc = tf.reshape( self.t_encoding(t), (-1, 28,28,1))
        
        in1 = tf.concat((img, t_enc), axis = 3)
        
        d1 = self.down_1(self.layer1(in1))
        d2 = self.down_2(self.layer2(d1))
        #d3 = self.down_3(self.layer3(d2))
        
        #u3 = self.up_3(d3)
        u2 = self.up_2(d2)
        u1 = self.up_1( tf.concat((u2, d1), axis = 3) )
        
        out = self.last_conv(tf.concat( (u1,in1), axis = 3) )
        
        
        return out