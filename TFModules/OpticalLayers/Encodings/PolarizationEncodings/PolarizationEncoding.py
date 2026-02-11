import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
from .CircularEncoding import CircularEncoding
from .LinearEncoding import LinearEncoding
from .LinearIntensityEncoding import LinearIntensityEncoding

class PolarizationEncoding(OpticalLayer):
    '''
    Bundle class for all jones polarization encodings
    '''
    def __init__(self, encoding = 'linear', **kwargs):
        super(PolarizationEncoding,self).__init__(**kwargs)
        self.encoding = encoding
        if encoding == 'linear':
            self.layer = LinearEncoding()
        elif encoding == 'circular':
            self.layer = CircularEncoding()
        elif encoding == 'linear_intensity_x':
            self.layer = LinearIntensityEncoding('x')
        elif encoding == 'linear_intensity_y':
            self.layer = LinearIntensityEncoding('y')
        elif encoding == 'linear_intensity_xy':
            self.layer = LinearIntensityEncoding('xy')
        else:
            raise ValueError()

    def call(self,input):
        return self.layer(input)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def get_config(self):
        temp = {
            'encoding': self.encoding
        }
        return temp

