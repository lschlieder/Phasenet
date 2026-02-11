import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
import PhaseplateNetwork.TFModules.OpticalLayers.Encodings.PolarizationEncodings as PE
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN


# from PhaseplateNetwork.TFModules.OpticalLayers.LinearPolarizationFilter import LinearPolarizationFilter
class NikileshSetupVariableDistance(DDNN):
    def __init__(self, use_hologram = False, nonlinearity = False, distance = 0.04, plates = 4, rotation_angle = np.pi/2, trainable_pixel=112, plate_scale_factor=2,
                 propagation_size=0.001792, propagation_pixel=224, padding=0.0015, frequency=3.843e14, wavespeed=3e8, mode ='ortho', input_mode = 'linear',
                 **kwargs):
        super(NikileshSetupVariableDistance, self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel,
                                               padding, frequency, wavespeed, **kwargs)
        input_shape = [None, self.propagation_pixel, self.propagation_pixel, 1]
        input_shape = self.append_element(self.get_padding_layer((input_shape[1], input_shape[2])), input_shape)
        #input_shape = self.append_element(OL.AmplitudeToIntensityJonesPolarization(axis = 'xy'),input_shape)
        input_shape = self.append_element(PE.PolarizationEncoding(input_mode), input_shape)
        #input_shape = self.append_element(OL.AmplitudeToRotation(), input_shape)
        input_shape = self.append_element(OL.AmplitudeToRotation(),input_shape)
        input_shape = self.append_element(self.get_wave_propagation(distance, input_shape[1]),input_shape)
        input_shape = self.append_element(OL.MultiplyWithSuperGaussian(input_shape[1:3]), input_shape)

        for i in range(0,plates):
            #input_shape = self.append_element(self.get_jones_phaseplate(False, True, 'x'),input_shape)
            input_shape = self.append_element(self.get_wave_propagation(distance, input_shape[1]), input_shape)
            input_shape = self.append_element(OL.MultiplyWithSuperGaussian(input_shape[1:3]), input_shape)
            if use_hologram:
                input_shape = self.append_element(self.get_jones_phaseplate(False, True, 'x'), input_shape)
            else:
                input_shape = self.append_element(self.get_rotating_plate(), input_shape)
            #input_shae = self.append_element(self.get_jones_phaseplate(False, True,'xy'), input_shape)
            input_shape = self.append_element(self.get_wave_propagation(distance, input_shape[1]),input_shape)
            input_shape = self.append_element(OL.MultiplyWithSuperGaussian(input_shape[1:3]), input_shape)

            if nonlinearity:
                print('adding nonlinearity')
                #input_shape = self.append_element(OL.LinearPolarizationFilter(angle = pol_angle), input_shape)
                #input_shape = self.append_element(OL.AmplitudeToRotation(),input_shape)
                input_shape = self.append_element(OL.PolarizationRotationNonlinearity(mode, rotation_angle = rotation_angle), input_shape)
            #input_shape = self.append_element(OL.AbsFromJonesPolarizationLayer(), input_shape)
            #input_shape = self.append_element(OL.AmplitudeToPolarizationAngle(np.pi/2), input_shape)

        #input_shape = self.append_element(self.get_rotating_plate(), input_shape)
        input_shape = self.append_element(self.get_wave_propagation(distance, input_shape[1]), input_shape)
        input_shape = self.append_element(OL.MultiplyWithSuperGaussian(input_shape[1:3]), input_shape)
        input_shape = self.append_element(OL.LinearPolarizationFilter(angle=0.0), input_shape)
        input_shape = self.append_element(OL.AbsFromJonesPolarizationLayer(),input_shape)
        output_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)

        return
