import numpy as np
import tensorflow as tf
from PhaseplateNetwork.TFModules.OpticalLayers.PhasePlate import PhasePlate
import matplotlib.pyplot as plt


class MultilenseArrayPhasePlate(PhasePlate):
    def __init__(self ,shape, L, scale = 2, wavelength= 0.0001, focus = 0.01, multi = 4,  **kwargs):
        super(MultilenseArrayPhasePlate,self).__init__(shape, scale, False,False, **kwargs)

        #self.N = shap * scale

        self.dp = L/np.array(shape)
        shape = np.array(shape)/multi

        [X, Y] = np.meshgrid((np.arange(-shape[0] // 2 + 1, shape[0] // 2 + 1, 1)) * self.dp[0],
                             (np.arange(-shape[1] // 2 + 1, shape[1] // 2 + 1, 1)) * self.dp[1])
        X_squared = X ** 2
        Y_squared = Y ** 2
        #print(wavelength)
        #print(focus)
        #print(L)
        #print(self.dp)
        phase = -(np.pi / (wavelength * focus) * (X_squared + Y_squared)).astype('float32') % (2 * np.pi) - np.pi
        phase = tf.tile(phase,[multi,multi])
        #phase = 0.5 * (np.log((1 + phase / np.pi) / (1 - phase / np.pi)))

        plt.imshow(phase)
        plt.show()
        self.phases.assign( tf.reshape(phase, self.phaseplate_shape))

        #self.phases = tf.Variable(tf.reshape(phase,self.phaseplate_shape), trainable= False,dtype = tf.float32)