import tensorflow as tf
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class UpSampling2D(OpticalLayer):
    def __init__(self, factor = (2,2),interpolation = "nearest", **kwargs):
        super(UpSampling2D,self).__init__(**kwargs)
        #self.factor = factor
        #self.average_pool = tf.keras.layers.AveragePooling2D(pool_size = factor, strides = strides)
        self.upsample = tf.keras.layers.UpSampling2D(size = factor,interpolation = interpolation)
    
    def call(self, input):
        inp_amp = tf.math.abs(input)
        inp_angle = tf.math.angle(input)

        up_amp = self.upsample(inp_amp)
        up_angle = self.upsample(inp_amp)
        res = tf.complex(up_amp,0.0)*tf.math.exp(1j*tf.complex(up_angle,0.0))
        return res

    def compute_output_shape(self, input_shape):
        return self.upsample.compute_output_shape(input_shape) 
