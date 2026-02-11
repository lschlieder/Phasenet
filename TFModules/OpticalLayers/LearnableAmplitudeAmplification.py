import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer  as OL

class LearnableAmplitudeAmplification(OL):
    def __init__(self , min = 0.0, max = None, amplitude_trainable= True, phase_trainable = False, use_regularizer = False, lam = 0.1, **kwargs):
        super(LearnableAmplitudeAmplification, self).__init__(**kwargs)

        if max != None:
            constraint = tf.keras.constraints.MinMaxNorm(min = min, max = max)
        else:
            constraint = None

        if use_regularizer != None:
            def offset_l1(wm, offset = 1.0, lam = 1.0):
                return lam * tf.reduce_sum(tf.abs( wm - offset))
            def offset_l2(wm, offset = 1.0, lam = 1.0):
                return lam * tf.reduce_sum(tf.square(wm - offset))
            
            amp_regularizer = lambda x: offset_l2(x, 1.0, lam)
            phase_regularizer = lambda x: offset_l2(x, 0.0,lam)
            #regularizer = 
        else:
            amp_regularizer = None 
            phase_regularizer = None

        self.amp = self.add_weight( name = 'learnable_amplitude_amplification', shape = (1), dtype = tf.float32, 
                                   initializer = tf.keras.initializers.constant(1.0), 
                                   trainable = amplitude_trainable,
                                   constraint = constraint, 
                                   regularizer = amp_regularizer
                                   )
        self.phase = self.add_weight( name = 'learnable_amplitude_amplification', shape = (1), dtype = tf.float32,
                                      initializer = tf.keras.initializers.constant(0.0), 
                                      trainable = phase_trainable,
                                      constraint = constraint,
                                      regularizer = phase_regularizer
                                      )



    def call(self, input):
        return input * tf.complex(self.amp, 0.0) * tf.math.exp( tf.complex(0.0, self.phase))

    def compute_output_shape(self, input_shape):
        return input_shape