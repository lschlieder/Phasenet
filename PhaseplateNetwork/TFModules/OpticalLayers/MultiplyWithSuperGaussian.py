import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class MultiplyWithSuperGaussian(OpticalLayer):
    def __init__(self,shape = [60,60], sg = 0.7, power = 2, offset = 0.2,**kwargs):
        super(MultiplyWithSuperGaussian,self).__init__(**kwargs)
        self.sg = sg
        self.power = power
        self.offset = offset
        self.shape = shape
        self.sgz = self.super_gaussian([sg,power,offset], shape)

    def super_gaussian(self, sg_params, N):
        sc = sg_params[0]
        p = sg_params[1]
        t = sg_params[2]
        x = np.linspace(-N[1] / 2, N[1] / 2 - 1, N[1])
        y = np.linspace(-N[0] / 2, N[0] / 2 - 1, N[0])
        xv, yv = np.meshgrid(x, y)
        sg = (1 - t) * np.exp(-np.power(xv / (N[1] * sc), p)) * np.exp(-np.power(yv / (N[0] * sc), p)) + t

        sgz = tf.cast(tf.constant(sg), dtype = tf.complex64)
        #print(sgz.shape)
        sgz = tf.reshape(sgz,(1,N[0],N[1],1))
        return sgz

    def call(self, Input, **kwargs):
        super_gaussian = tf.tile(self.sgz,(1, 1,1, Input.shape[3]))
        return Input * super_gaussian

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2],input_shape[3]]

    def get_config(self):
        temp = {
            'sg': self.sg,
            'power': self.power,
            'offset': self.offset,
            'shape': self.shape
        }
        return temp