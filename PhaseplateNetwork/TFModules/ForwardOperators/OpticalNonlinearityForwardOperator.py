import tensorflow as tf
from PhaseplateNetwork.TFModules.ForwardOperators.ForwardOperator import ForwardOperator
from PhaseplateNetwork.TFModules.ForwardOperators.WavePropagation import WavePropagation
import warnings

class OpticalNonlinearityForwardOperator(ForwardOperator):
    '''
    An optical element that applies a nonlinearity to the field in a regime of size depth with a resolution of resolution. Keep in mind
    a high resolution will lead to slow code, since the number of propagation calculations is (Depth/resolution)
    '''
    def __init__(self, Depth = 10, resolution = 1, N=100, L=80, scale = 2, padding=5, f0=2.00e6, cM=1484, use_FT=True, jitter=False, activation_fn = None, inverse_activation_fn = None, **kwargs):
        super(OpticalNonlinearityForwardOperatorForwardOperator, self).__init__(**kwargs)
        self.Depth = Depth
        self.resolution = resolution
        self.NPropagations =Depth / resolution
        if self.NPropagations.is_integer():
            self.NPropagations = int(self.NPropagations)
        else:
            self.NPropagations = int(self.NPropagations)
            warnings.warn("Depth is not clearly divisible by resolution. Continuing with {} propagations for a total element size of {}".format(self.NPropagations, self.resolution * self.NPropagations), Warning)
        self.propagations = []
        for i in range(0,self.NPropagations):
            self.propagations.append(WavePropagation(resolution, N, L,scale,  padding, f0, cM, True, use_FT))

        self.activation_fn = activation_fn
        self.inverse_activation_fn = inverse_activation_fn

    def call(self,Input):

        u = Input
        for i in range(0, self.NPropagations):
            u = self.activation_fn(self.propagations[i].call(u))
        return u

    def inverse_call(self, Input):
        u = Input
        for i in range(self.NPropagations,0 ,-1):
            u = self.inverse_activation_fn(self.propagations[i].inverse_call(u))
        return u

