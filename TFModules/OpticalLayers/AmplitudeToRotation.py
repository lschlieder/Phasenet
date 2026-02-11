import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
from PhaseplateNetwork.TFModules.OpticalLayers.AbsFromJonesPolarizationLayer import AbsFromJonesPolarizationLayer
#from PhaseplateNetwork.TFModules.OpticalLayers
class AmplitudeToRotation(OpticalLayer):
    def __init__(self, max_angle = np.pi/2, offset = 0.0,**kwargs):
        super(AmplitudeToRotation,self).__init__(**kwargs)
        self.max_angle = max_angle
        self.abs_layer = AbsFromJonesPolarizationLayer()
        self.offset = offset


    def get_rotation_matrix(self, angle):
        angle = tf.cast(-angle, dtype = tf.float32)
        rotation_mat_up = tf.concat((tf.math.cos(angle), -tf.math.sin(angle)), axis=3)
        rotation_mat_down = tf.concat((tf.math.sin(angle), tf.math.cos(angle)), axis=3)
        rotation_mat = tf.stack((rotation_mat_up, rotation_mat_down), axis=4)
        return tf.cast(rotation_mat, dtype = tf.complex64)

    def call(self,input, rotation_amplitude = None, **kwargs):
        if rotation_amplitude == None:
            abs_jones = tf.tile(self.abs_layer(input), (1, 1, 1, 1))
        else:
            abs_jones = tf.tile(self.abs_layer(rotation_amplitude), (1,1,1,1))

        rotation_matrices = self.get_rotation_matrix(abs_jones * self.max_angle + self.offset)
        out = tf.linalg.matvec(rotation_matrices, input)
        return out

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2],2]

    def get_config(self):
        temp = {
            'max_angle': self.max_angle,
            'offset': self.offset
        }
        return temp

