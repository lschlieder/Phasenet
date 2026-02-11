import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.NetworkLayers as NL
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN
from PhaseplateNetwork.TFModules.Models.DDNNModels.JonesNetworks.NikileshSetup import NikileshSetup


# from PhaseplateNetwork.TFModules.OpticalLayers.LinearPolarizationFilter import LinearPolarizationFilter
class NikileshSetupNormalized(NikileshSetup):
    def __init__(self, use_hologram = False, nonlinearity = False, plates = 4, pol_angle = 0.0, trainable_pixel=112, plate_scale_factor=2,
                 propagation_size=0.001792, propagation_pixel=224, padding=0.0015, frequency=3.843e14, wavespeed=3e8,
                 **kwargs):
        print(plates)
        print(padding)
        print(propagation_pixel)
        super(NikileshSetupNormalized, self).__init__(use_hologram = use_hologram, nonlinearity = nonlinearity, plates = plates, pol_angle = pol_angle, trainable_pixel=trainable_pixel, plate_scale_factor=plate_scale_factor,
                 propagation_size=propagation_size, propagation_pixel=propagation_pixel, padding=padding, frequency=frequency, wavespeed=wavespeed,
                 **kwargs)
        print(self.out_s)
        #self.out_s = self.append_element(NL.AbsLayer(), self.out_s)
        #self.out_s = self.append_element(NL.NormalizingLayer(), self.out_s)
        self.normalizing_layer = NL.MultiplyingNormalizingLayer()
        return

    def call(self, input):
        out = super(NikileshSetupNormalized, self).call(input)
        return self.normalizing_layer(tf.math.abs(out))