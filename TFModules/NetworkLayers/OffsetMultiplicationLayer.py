import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer  as OL

class OffsetMultiplicationLayer(tf.keras.layers.Layer):
    def __init__(self , min = 0.0, max = None, trainable= True, use_regularizer = False, offset = 0.0, lam = 0.1, **kwargs):
        super(OffsetMultiplicationLayer, self).__init__(**kwargs)

        if max != None:
            constraint = tf.keras.constraints.MinMaxNorm(min = min, max = max)
        else:
            constraint = None

        if use_regularizer:
            def offset_l1(wm, offset = 1.0, lam = 1.0):
                return lam * tf.reduce_sum(tf.abs( wm - offset))
            def offset_l2(wm, offset = 1.0, lam = 1.0):
                return lam * tf.reduce_sum(tf.square(wm - offset))
            
            regularizer = lambda x: offset_l2(x, offset, lam)

        else:
            regularizer = None 
            
        self.scale_var = self.add_weight( name = 'learnable_scale_var', shape = (), dtype = tf.float32, 
                                   initializer = tf.keras.initializers.constant(1.0), 
                                   trainable = trainable,
                                   constraint = constraint, 
                                   regularizer = regularizer
                                   )



    def call(self, input):
        return input * self.scale_var

    def compute_output_shape(self, input_shape):
        return input_shape