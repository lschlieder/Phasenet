import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
#import PhaseplateNetwork.TFModules.Models.DDNNModels as Models
#import PhaseplateNetwork.TFModules.Models.DDNNModels.JonesNetworkWithoutInput
from PhaseplateNetwork.TFModules.Models.DDNNModels.RecurrentDDNN import RecurrentDDNN
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN
import importlib
import matplotlib.pyplot as plt

class SRLatch(RecurrentDDNN):
    def __init__(self, NetworkModel="JonesNetworkWithoutInput", num_iter=4, trainable_pixel=112, plate_scale_factor=2,
                 propagation_size=0.001792, propagation_pixel=224, padding=0.00175, frequency=3.843e14, wavespeed=3e8,
                 **kwargs):
        super(SRLatch, self).__init__(NetworkModel = NetworkModel, num_iter = num_iter, trainable_pixel=trainable_pixel, plate_scale_factor=plate_scale_factor,
                                            propagation_size=propagation_size, propagation_pixel=propagation_pixel,
                                            padding=padding, frequency=frequency, wavespeed=wavespeed, **kwargs)

        self.PolFiltery = OL.LinearPolarizationFilter(angle = np.pi/2)
        self.PolFilterx = OL.LinearPolarizationFilter(angle = 0.0)



    #def recurrent_loss(self, input, recurrent):
    #    return mse(input, recurrent)

    def call(self,Input):
        #inp = self.Input_layer(Input)

        assert (Input.shape[3] == 2)
        inp = tf.cast(Input, dtype = tf.complex64)

        #out_arr = []
        #inp_arr = []
        for i in range(0, self.num_iter):
            output = self.forward_diffraction(inp)
            re = self.backward_diffraction(self.PolFiltery(output))
            inp = tf.concat((inp[:,:,:,0:1], re[:,:,:,1:2]), axis = 3)
            #out_arr.append(output[:,:,:,0:1])
            #inp_arr.append(re[:,:,:,1:2])
        #self.add_loss(self.recurrent_loss(output[:,:,:,0:1], re[:,:,:,1:2]))
        out = tf.concat((output[:,:,:,0:1], re[:,:,:,1:2]), axis = 3)
        return out

    def get_propagation_fields(self, Input):
        #inp = self.Input_layer(Input)
        inp = tf.cast(Input,dtype = tf.complex64)
        for i in range(0, self.num_iter-1):
            output = self.forward_diffraction(inp)
            re = self.backward_diffraction(self.PolFiltery(output))
            inp = tf.concat((inp[:,:,:,0:1], re[:,:,:,1:2]), axis = 3)
        #output = self.forward_diffraction(inp)
        #re = self.backward_diffraction(output[:,:,:,0:1])

        forward = self.forward_diffraction.get_propagation_fields(inp)
        output = self.forward_diffraction(inp)
        backward = self.backward_diffraction.get_propagation_fields(self.PolFiltery(output))

        #self.save_fields([inp]+forward+backward+[output[:,:,:1:2]], PATH, False)
        return [inp.numpy()]+forward+backward+[output[:,:,:1:2].numpy()]
