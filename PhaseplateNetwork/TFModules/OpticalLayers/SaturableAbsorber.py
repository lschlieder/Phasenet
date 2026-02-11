import tensorflow as tf
from PhaseplateNetwork.TFModules.OpticalLayers.WavePropagation import WavePropagation
import warnings
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalNonlinearity import OpticalNonlinearity
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
import matplotlib.pyplot as plt

class SaturableAbsorber(OpticalNonlinearity):
    def __init__(self, depth = 0.01, resolution = 0.01, N=100, L=0.08, padding=0.05, f0=2.00e6, cM=1484, use_FT=True, OD = 1.0, I_saturation = 0.1, I_sat_trainable = False, use_bias = False, **kwargs):

        I_sat = tf.Variable(I_saturation, trainable= False)


        #@tf.custom_gradient
        #def saturable_absorber_fn(x):
        #    re = x* tf.math.exp( -OD/(1+x**2/I_sat))
        #    def grad(dy):
        #        return dy * (1 + ((x**2)/I_sat)*( OD/ (1+(x**2)/I_sat)**2 )) * tf.math.exp(-(OD/2) / (1 + (x**2) / I_sat))
        #    return re, grad


        self.OD = OD
        self.I_saturation = I_saturation
        def sat_abs_fun(x):
            #amp = saturable_absorber_fn(tf.math.abs(x))
            #print(x)
            amp = tf.math.abs(x) * tf.math.exp( -(OD/2)/(1+tf.math.abs(x)**2/I_sat))
            phase = tf.math.exp(1j* tf.cast(tf.math.angle(x), dtype = tf.complex64))
            #plt.imshow(np.abs(tf.cast(amp,dtype = tf.complex64)*phase)[0,:,:,0])
            #plt.figure()
            #plt.imshow(np.abs(tf.cast(amp,dtype = tf.complex64)*phase)[0,:,:,1])
            #plt.show()
            #input()
            return tf.cast(amp,dtype = tf.complex64) * phase

        super(SaturableAbsorber,self).__init__(depth, resolution , N, L, sat_abs_fun, padding, f0, cM, use_FT,  **kwargs)


    def get_config(self):
        temp = {
            "Depth": self.Depth,
            "resolution":self.resolution,
            "N": self.N,
            "L": self.L,
            "padding": self.padding,
            "f0": self.f0,
            "cM": self.cM,
            "use_FT": self.use_FT,
            "OpticalDensity": self.OD,
            "SaturationIntensity": self.I_saturation
        }
        return temp

    @classmethod
    def from_config(cls, config):
        el = cls( Depth = config['Depth'], resolution = config['resolution'], N=config['N'], L=config['L'],
            padding=config['padding'], f0=config['f0'], cM=config['cM'], use_FT=config['use_FT'],
            OD = config['OpticalDensity'], I_saturation = config['SaturationIntensity'])
        return el