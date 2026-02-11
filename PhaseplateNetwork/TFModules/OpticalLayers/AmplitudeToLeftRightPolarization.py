import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class AmplitudeToLeftRightPolarization(OpticalLayer):
    def __init__(self, **kwargs):
        super(AmplitudeToLeftRightPolarization,self).__init__(**kwargs)

        self.right = tf.constant( 1/np.sqrt(2) * np.array([1, -1j]), dtype = tf.complex64)
        self.left = tf.constant( 1/np.sqrt(2) * np.array([1, 1j]), dtype = tf.complex64)

    def call(self,input):
        abs = tf.math.abs(input)
        phase = tf.math.angle(input)

        abs = abs/tf.reshape(tf.math.reduce_max(abs, axis = (1,2,3)),(tf.shape(abs)[0],1,1,1))
        #print(abs.shape)
        #print(tf.reshape(tf.math.reduce_max(abs, axis = (1,2,3)),(tf.shape(abs)[0],1,1,1)).shape)
        output = tf.complex((1-abs),0.0)*tf.math.exp(1j*tf.complex(phase,0.0)) * self.left + tf.complex(abs,0.0)*tf.math.exp(1j*tf.complex(phase,0.0)) * self.right
        #x_pol = tf.complex(tf.math.cos(abs*np.pi/2.0),0.0) * tf.math.exp(1j* tf.complex(phase,0.0))
        #y_pol = tf.complex(tf.math.sin(abs*np.pi/2.0),0.0)* tf.math.exp(1j * tf.complex(phase,0.0))

        return output

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2],2]

    def get_config(self):
        temp = {
        }
        return temp

