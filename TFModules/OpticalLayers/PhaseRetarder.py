import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class PhaseRetarder(OpticalLayer):
    '''
    Phase retarded
    '''
    def __init__(self, diff = np.pi/2, rot_angle = 0, circularity = np.pi/2,**kwargs):
        '''
        Creates an arbitrary phase retarder
        diff: difference between the angles of the fast and slow axis
        rot_angle: rotation between the fast axis and the x axis
        circularity: Note that for linear retarders, ϕ = 0 and for circular retarders, ϕ = ± π/2, θ = π/4. In general for elliptical retarders, ϕ takes on values between - π/2 and π/2.
        '''
        super(PhaseRetarder, self).__init__(**kwargs)
        self.diff = diff
        self.rot_angle = rot_angle
        self.circularity = circularity
        x11 = np.cos(rot_angle)**2+ np.exp(1j*diff)*np.sin(rot_angle)**2
        x12 = (1-np.exp(1j*diff))*np.exp(-1j*circularity)*np.cos(rot_angle)*np.sin(rot_angle)
        x21 = (1- np.exp(1j*diff))*np.exp(1j*circularity)*np.cos(rot_angle)*np.sin(rot_angle)
        x22 = np.sin(rot_angle)**2 + np.exp(1j*diff)*np.cos(rot_angle)**2

        pol_mat = tf.math.exp(-1j*diff/2)* np.array([[x11, x12],
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
            'direction': self.direction
        }
        return temp

