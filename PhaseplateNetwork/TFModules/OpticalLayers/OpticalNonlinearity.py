import tensorflow as tf
from PhaseplateNetwork.TFModules.OpticalLayers.WavePropagation import WavePropagation
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
import warnings
import numpy as np

class OpticalNonlinearity(OpticalLayer):
    '''
    An optical element that applies a nonlinearity to the field in a regime of size depth with a resolution of resolution. Keep in mind
    a high resolution will lead to slow code, since the number of propagation calculations is (Depth/resolution)
    :param Depth: the size of the optical linearity in mm
    :param resolution: the resolution in propagation direction. Increase this for increasing self interaction effects.
    :param N: The number of pixels
    :param L: The physical size of the propagation image in mm
    :param scale: the upscaling factor for the propagation (higher is better, but quickly gets too complicated)
    :param activation_fn: backpropagatable activation function
    :param padding: padding in mm for the propagation
    :param f0: frequency in 1/s
    :param cM: Wave speed in medium im mm/s
    :param use_FT: Should the fourier transform be used
    '''
    def __init__(self, Depth = 10, resolution = 1, N=100, L=80, activation_fn = None, padding = None, f0=2.00e6, cM=1484, use_FT=True,  **kwargs):
        super(OpticalNonlinearity, self).__init__(**kwargs)
        self.Depth = Depth
        self.resolution = resolution
        self.NPropagations =Depth / resolution
        self.N = N
        self.L = L
        self.padding = padding
        self.use_FT = use_FT
        self.f0 = f0
        self.cM = cM

        if self.NPropagations.is_integer():
            self.NPropagations = int(self.NPropagations)
        else:
            self.NPropagations = int(np.ceil(self.NPropagations))
            warnings.warn("Depth is not clearly divisible by resolution. Continuing with {} propagations for a total element size of {}".format(self.NPropagations, self.resolution * self.NPropagations), Warning)
        self.propagations = []
        print(self.NPropagations)
        for i in range(0,self.NPropagations):
            self.propagations.append(WavePropagation(resolution, N, L, padding, f0, cM, True, use_FT))

        self.activation_fn = activation_fn

    def call(self,Input):

        u = Input
        for i in range(0, self.NPropagations):
            u = self.activation_fn(self.propagations[i].call(u))
        return u

    def compute_output_shape(self, input_shape):
        return input_shape



