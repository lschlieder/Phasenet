import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN
import PhaseplateNetwork.TFModules.OpticalLayers.Encodings.PolarizationEncodings as PE

# from PhaseplateNetwork.TFModules.OpticalLayers.LinearPolarizationFilter import LinearPolarizationFilter
class NikileshPaper(DDNN):
    def __init__(self, plate_mode = 'rotation_clamped', max_angle = np.pi/2, propagation_distance = 0.2, trainable_pixel=400, plate_scale_factor=1,
                 propagation_size=0.0144, propagation_pixel=400, padding=0.0144, frequency=5.639097744e14, wavespeed=3e8,
                 **kwargs):
        super(NikileshPaper, self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel,
                                               padding, frequency, wavespeed, **kwargs)
        input_shape = [None, self.propagation_pixel, self.propagation_pixel, 1]
        input_shape = self.append_element(self.get_padding_layer((input_shape[1], input_shape[2])), input_shape)
        input_shape = self.append_element(PE.LinearIntensityEncoding('xy'), input_shape)
        #input_shape = self.append_element(self.get_wave_propagation(0.1, input_shape[1]),input_shape)
        input_shape = self.append_element(OL.PolarizationRotator(angle = np.pi/4), input_shape)
        #input_shape = self.append_element(OL.MultiplyWithSuperGaussian(input_shape[1:3]), input_shape)

        print(plate_mode)
        if plate_mode == 'rotation_clamped':
            input_shape = self.append_element(self.get_rotating_plate_constraint(max_angle), input_shape)
        elif plate_mode == 'rotation':
            input_shape = self.append_element(self.get_rotating_plate(), input_shape)
        elif plate_mode == 'phase_x':
            input_shape = self.append_element(self.get_jones_phaseplate(False, True, 'x'), input_shape)
        elif plate_mode == 'phase_y':
            input_shape = self.append_element(self.get_jones_phaseplate(False, True, 'y'), input_shape)    
        elif plate_mode == 'phase_xy':
            input_shape = self.append_element(self.get_jones_phaseplate(False,True, 'xy'),input_shape)
        elif plate_mode == 'phase_x=y':
            input_shape = self.append_element(self.get_phaseplate(False,True),input_shape)
               
        #input_shape = self.append_element(self.get_jones_phaseplate(False, True, 'x'), input_shape)
        input_shape = self.append_element(self.get_wave_propagation(propagation_distance, input_shape[1]), input_shape)
        input_shape = self.append_element(OL.AbsFromJonesPolarizationLayer(),input_shape)
        input_shape = self.append_element(OL.AbsLayer(), input_shape)
        self.out_s = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)


        return
