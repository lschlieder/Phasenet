import tensorflow as tf
import numpy as np


class FourierForwardConstraint(Layers.ForwardOperators.ForwardOperator):
    def __init__(self ):
        super(FourierForwardConstraint,self).__init__()


    @tf.function
    def call(self,input):
        return tf.signal.fft2d( input)

    @tf.function
    def call(self, input):
        return tf.signal.ifft2d( input)