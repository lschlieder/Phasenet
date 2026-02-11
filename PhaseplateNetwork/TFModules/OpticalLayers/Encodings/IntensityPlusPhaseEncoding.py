from .IntensityEncoding import IntensityEncoding
from .PhaseEncoding import PhaseEncoding
import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class IntensityPlusPhaseEncoding(OpticalLayer):
    def __init__(self, output_size, sigma = 0.5,**kwargs):
        super(IntensityPlusPhaseEncoding,self).__init__(**kwargs)
        self.phase_enc = PhaseEncoding()
        self.int_enc = IntensityEncoding()


        self.output_size = output_size
        self.szIm = int(output_size/2)
        #szIm = 56
        x, y = np.meshgrid(np.linspace(-1,1,self.szIm), np.linspace(-1,1,self.szIm))
        d = np.sqrt(x*x+y*y)
        sigma, mu = sigma, 0.0
        g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        self.sIm = self.szIm
        self.gaussian = tf.constant(g, dtype = tf.float32)

    def call(self,input, **kwargs):
        inp_rescaled = tf.image.resize_with_pad(input, self.szIm, self.szIm)
        p = self.phase_enc(inp_rescaled)
        i = self.int_enc(inp_rescaled)

        g11 = tf.reshape(tf.complex(self.gaussian, 0.0),(1,self.sIm, self.sIm, 1)) * tf.math.exp(1j* tf.complex( tf.zeros_like(inp_rescaled), 0.0))
        g22 = tf.complex(tf.ones_like(inp_rescaled),0.0) * tf.reshape(tf.complex(self.gaussian, 0.0),(1,self.sIm, self.sIm, 1)) * tf.math.exp(1j* tf.reshape(tf.complex( self.gaussian*np.pi*2, 0.0),(1,self.sIm, self.sIm, 1) )  )

        top = tf.concat([g11,p] , axis = 2)
        bottom = tf.concat([i,g22], axis = 2)
        out = tf.concat([top,bottom], axis = 1)

        return out

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.output_size, self.output_size,1]

    def get_config(self):
        temp = {
        }
        return temp