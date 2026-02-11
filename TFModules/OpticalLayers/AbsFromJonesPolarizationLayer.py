import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class AbsFromJonesPolarizationLayer(OpticalLayer):
    def __init__(self, **kwargs):
        super(AbsFromJonesPolarizationLayer,self).__init__(**kwargs)

    def call(self,input, **kwargs):
        #sum = tf.expand_dims(tf.math.reduce_sum(input, axis = 3),axis = 3)
        sum = tf.expand_dims( tf.math.reduce_sum(tf.math.abs(input)**2, axis = 3),axis = 3)
        sum = tf.math.sqrt(sum)
        #print(sum.shape)
        return tf.cast(sum,dtype = tf.complex64)

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2],1]

    def get_config(self):
        temp = {
        }
        return temp