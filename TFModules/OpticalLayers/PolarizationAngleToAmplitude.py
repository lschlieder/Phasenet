import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class PolarizationAngleToAmplitude(OpticalLayer):
    def __init__(self, max_angle = np.pi, **kwargs):
        super(PolarizationAngleToAmplitude,self).__init__(**kwargs)
        self.max_angle = max_angle


    def get_rotation_matrix(self, angle):
        angle = -angle
        rotation_mat_up = tf.concat((tf.math.cos(angle), -tf.math.sin(angle)), axis=3)
        rotation_mat_down = tf.concat((tf.math.sin(angle), tf.math.cos(angle)), axis=3)
        rotation_mat = tf.stack((rotation_mat_up, rotation_mat_down), axis=4)
        return rotation_mat

    def call(self,input):
        abs = tf.math.abs(input)
        #phase = tf.math.angle(input)
        input_without_angle = tf.complex(abs,0.0)

        jones_start = tf.concat((tf.ones_like(input_without_angle), tf.zeros_like(input_without_angle)), axis = 3)

        rotation_matrices = self.get_rotation_matrix(input_without_angle*self.max_angle)
        #print(rotation_matrices.dtype)
        #print(input_without_angle.dtype)
        out = tf.linalg.matvec(rotation_matrices, jones_start)
        #abs = abs/tf.reshape(tf.math.reduce_max(abs, axis = (1,2,3)),(tf.shape(abs)[0],1,1,1))
        #output = tf.complex(abs,0.0)*tf.math.exp(1j*tf.complex(phase,0.0)) * self.x +
        #        tf.complex(tf.zeros_like(abs), 0.0) * tf.math.exp(1j * tf.complex(tf.zeros_like(phase), 0.0)) * self.y
        #output = tf.complex((1-abs),0.0)*tf.math.exp(1j*tf.complex(phase,0.0)) * self.x + tf.complex(abs,0.0)*tf.math.exp(1j*tf.complex(phase,0.0)) * self.y

        return out

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2],2]

    def get_config(self):
        temp = {
            'max_angle': self.max_angle
        }
        return temp

