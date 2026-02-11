import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN


# from PhaseplateNetwork.TFModules.OpticalLayers.LinearPolarizationFilter import LinearPolarizationFilter
class PolarizationNonlinearity(DDNN):
    def __init__(self, plate_per_nonlinearity = 2, use_hologram = False, nonlinearity = False, nonlinear_mode = 'ortho', plates = 4, rotation_angle = np.pi/2, trainable_pixel=112, plate_scale_factor=2,
                 propagation_size=0.001792, propagation_pixel=120, padding=0.0015, frequency=3.843e14, wavespeed=3e8,
                 **kwargs):
        super(PolarizationNonlinearity, self).__init__(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel,
                                               padding, frequency, wavespeed, **kwargs)
        input_shape = [None, self.propagation_pixel, self.propagation_pixel, 1]
        input_shape = self.append_element(self.get_padding_layer((input_shape[1], input_shape[2])), input_shape)
        input_shape = self.append_element(OL.AmplitudeToIntensityJonesPolarization(axis = 'xy'),input_shape)
        #input_shape = self.append_element(OL.AmplitudeToRotation(), input_shape)
        input_shape = self.append_element(OL.AmplitudeToRotation(),input_shape)
        input_shape = self.append_element(self.get_wave_propagation(0.04, input_shape[1]),input_shape)
        #input_shape = self.append_element(OL.MultiplyWithSuperGaussian(input_shape[1:3]), input_shape)

        num_plates = plates
        self.plates = []
        j = 0 
        for i in range(0,num_plates):
            #input_shape = self.append_element(self.get_jones_phaseplate(False, True, 'x'),input_shape)
            input_shape = self.append_element(self.get_wave_propagation(0.04, input_shape[1]), input_shape)
            #input_shape = self.append_element(OL.MultiplyWithSuperGaussian(input_shape[1:3]), input_shape)
            if use_hologram:
                p = self.get_jones_phaseplate(False, True, 'xy')
                #p._name = f'jones phaseplate {i}'
                input_shape = self.append_element(p, input_shape)
            else:
                p = self.get_rotating_plate()
                #p._name = f'rotating plate {i}'
                input_shape = self.append_element(p, input_shape)
            self.plates.append(p)
            #input_shae = self.append_element(self.get_jones_phaseplate(False, True,'xy'), input_shape)
            input_shape = self.append_element(self.get_wave_propagation(0.08, input_shape[1]),input_shape)
            #input_shape = self.append_element(OL.MultiplyWithSuperGaussian(input_shape[1:3]), input_shape)

            if nonlinearity and j == (plate_per_nonlinearity-1):
                print('adding nonlinearity')
                input_shape = self.append_element(OL.PolarizationRotationNonlinearity(nonlinear_mode, rotation_angle = rotation_angle), input_shape)
                j = 0 

            j +=1
            
        input_shape = self.append_element(self.get_rotating_plate(), input_shape)
        input_shape = self.append_element(self.get_wave_propagation(0.04, input_shape[1]), input_shape)
        #input_shape = self.append_element(OL.MultiplyWithSuperGaussian(input_shape[1:3]), input_shape)
        input_shape = self.append_element(OL.LinearPolarizationFilter(angle=0.0), input_shape)

        input_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)

        input_shape = self.append_element(OL.IntensityFromJonesPolarizationLayer(), input_shape)
        input_shape = self.append_element(OL.CategoryOutputLayer(10, (input_shape[1], input_shape[2]), [propagation_size, propagation_size],
                                                                    radius= propagation_size/10,
                                                                    distance = (propagation_size/10)*2, 
                                                                    output_shape = 'circle', 
                                                                    output_function = 'sum',
                                                                    negative_output = False),input_shape)
            
        self.out_s = input_shape
        #input_shape = self.append_element(OL.AbsFromJonesPolarizationLayer(),input_shape)
        #i#nput_shape = self.append_element(OL.AbsLayer(), input_shape)
  

        return
    
    def get_phaseplates(self):
        plates = []
        for c in self.plates:
            plates.append(c.get_image_variables())
        
        plates = np.squeeze(np.array(plates))
        return plates