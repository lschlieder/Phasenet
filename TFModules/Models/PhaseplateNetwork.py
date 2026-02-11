import tensorflow as tf
from PhaseplateNetwork.TFModules.ForwardOperators.MultiplateForwardOperator import MultiplateForwardOperator
import numpy as np

EPSILON = 1e-16
class PhaseplateNetwork(tf.keras.Model):
    def __init__(self, z, trainable_pixel=100, L=80, scale=2, padding=50, f0=2.00e6, cM=1484, use_FT=True, jitter=False, amplitude_trainable = False, phase_trainable = True,**kwargs):
        super(PhaseplateNetwork,self).__init__(**kwargs)
        self.z = z
        self.trainable_pixel = 100
        self.scale = 2
        self.L = L
        self.f0 = f0
        self.cM = cM
        self.use_FT = use_FT
        self.jitter = jitter
        self.Propagation = MultiplateForwardOperator(z, trainable_pixel=trainable_pixel, scale=scale, L=L, padding=padding, f0=f0, cM=cM, use_FT=use_FT, jitter=jitter)
        phaseplate_shape = self.Propagation.get_phaseshift_shape()
        self.amplitudes = tf.Variable(np.ones(phaseplate_shape),trainable = amplitude_trainable, dtype = tf.float32)
        self.phases = tf.Variable(np.zeros(phaseplate_shape), trainable= phase_trainable,dtype = tf.float32)
        #self.phase_plates = tf.complex(self.amplitudes+EPSILON,0.0) * tf.cast( tf.exp( 1j * tf.cast(self.phases, dtype = tf.complex64)), dtype = tf.complex64)

    def call(self,Input):
        #phase_plates = tf.complex(self.amplitudes+EPSILON,0.0) * tf.cast(tf.exp( 1j*self.phases), dtype = tf.complex64)
        Input = tf.cast(Input,dtype = tf.complex64)
        self.phase_plates = tf.complex(self.amplitudes+EPSILON,0.0) * tf.cast( tf.exp( 1j * tf.cast(self.phases, dtype = tf.complex64)), dtype = tf.complex64)
        output = self.Propagation( Input, self.phase_plates)

        return output

    def get_image_variables(self):
        ret = []
        for i in range(0,len(self.trainable_variables)):
            for j in range(0,self.trainable_variables[i].shape[0]):
                ret.append( self.trainable_variables[i][j,:,:])
        return ret


