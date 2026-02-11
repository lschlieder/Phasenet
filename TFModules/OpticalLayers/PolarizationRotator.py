import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class PolarizationRotator(OpticalLayer):
    '''
    Polarization Rotator
    '''
    def __init__(self, angle = np.pi/2,**kwargs):
        '''
        Creates an Optical rotation device, that rotates the incoming jones vector by param:angle rad
        '''
        super(PolarizationRotator, self).__init__(**kwargs)
        self.angle = angle

        x11 = np.cos(angle)
        x12 = -np.sin(angle)
        x21 = np.sin(angle)
        x22 = np.cos(angle)

        pol_mat =  np.array([[x11, x12],
                             [x21, x22]])

        self.pol_mat = tf.cast(tf.reshape(pol_mat, (1, 1, 2, 2)), dtype = tf.complex64)

    def call(self,input):
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

