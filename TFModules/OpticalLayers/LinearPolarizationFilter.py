import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class LinearPolarizationFilter(OpticalLayer):
    '''
    Linear Polarization Filter for Jones calculus images.
    '''
    def __init__(self, angle = np.pi/4,**kwargs):
        '''
        Creates the Linear polarization filter
        angle: Angle of the polarization filter in rad
        '''
        super(LinearPolarizationFilter, self).__init__(**kwargs)
        self.angle = angle

        pol_mat = np.array([[np.cos(angle)**2, np.sin(angle)*np.cos(angle)],
                            [np.sin(angle)*np.cos(angle), np.sin(angle)**2]])
        self.pol_mat = tf.cast(tf.reshape(pol_mat, (1, 1, 2, 2)), dtype = tf.complex64)

    def call(self,input, **kwargs):
        assert(input.shape[3] == 2)
        return tf.linalg.matvec(self.pol_mat, input)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(0,len(input_shape)):
            output_shape.append(input_shape[i])
        #output_shape[3] = 2
        return output_shape

    def get_config(self):
        #shape, scale = 2, amplitude_trainable = False, phase_trainable = True
        temp = {
            'angle': self.angle
        }
        return temp

