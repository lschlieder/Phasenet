import tensorflow as tf
from PhaseplateNetwork.TFModules.OpticalLayers.WavePropagation import WavePropagation
import warnings
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalNonlinearity import OpticalNonlinearity
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

EPSILON = 0.00000000000001

class ReluNonlinearity(OpticalNonlinearity):
    def __init__(self, Depth = 0.01, resolution = 0.01, N=100, L=80, padding=5, f0=2.00e6, cM=1484, use_FT=True, OD = 1.0, I_sat_trainable = False, use_bias = False, **kwargs):
                      # (z, 0.01, n_p, L, 1, padding, f0, cM, use_FT)
        Bias = tf.Variable(0.02, trainable= False)

        def relu_fun(x):
            return tf.cast(tf.nn.relu(tf.abs(x)+EPSILON-Bias), dtype = tf.complex64) * tf.math.exp(1j*tf.cast(tf.math.angle(x), dtype = tf.complex64))
        super(ReluNonlinearity,self).__init__(Depth, resolution , N, L , relu_fun, padding, f0, cM, use_FT,  **kwargs)