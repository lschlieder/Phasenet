import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
from PhaseplateNetwork.TFModules.OpticalLayers.LinearPolarizationFilter import LinearPolarizationFilter
from PhaseplateNetwork.TFModules.OpticalLayers.AmplitudeToRotation import AmplitudeToRotation

class PolarizationRotationNonlinearity(OpticalLayer):
    def __init__(self, rotation_pol_direction = 'same', pol_angle = 0.0, rotation_angle = np.pi/2, offset = 0.0, **kwargs):
        super(PolarizationRotationNonlinearity,self).__init__(**kwargs)
        self.angle = pol_angle
        self.rotation_angle = rotation_angle
        self.offset = offset
        self.main_pol = LinearPolarizationFilter(angle = pol_angle)
        self.ortho_pol = LinearPolarizationFilter(angle = pol_angle + np.pi/2)
        self.rotation_pol_direction = rotation_pol_direction
        self.AmpToRotation = AmplitudeToRotation(rotation_angle, offset)

    def call(self, input, **kwargs):
        assert(input.shape[3] == 2)
        main_polarized = self.main_pol(input)
        ortho_polarized = self.ortho_pol(input)
        if self.rotation_pol_direction == 'same':
            output = self.AmpToRotation(main_polarized, main_polarized)
        elif self.rotation_pol_direction == 'ortho':
            output = self.AmpToRotation(main_polarized, ortho_polarized)

        return output

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2],2]

    def get_config(self):
        temp = {
            'angle': self.angle,
            'rotation_pol_direction': self.rotation_pol_direction
        }
        return temp



