from multiprocessing.dummy import Pool
import tensorflow as tf
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class PoolingLayer(OpticalLayer):
    def __init__(self, factor = (2,2),strides = None, **kwargs):
        super(PoolingLayer,self).__init__(**kwargs)
        #self.factor = factor
        self.average_pool = tf.keras.layers.AveragePooling2D(pool_size = factor, strides = strides)

    def call(self, input):
        inp = tf.math.abs(input)
        return self.average_pool(inp)

    def compute_output_shape(self, input_shape):
        return self.average_pool.compute_output_shape(input_shape) 
