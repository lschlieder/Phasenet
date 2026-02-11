import tensorflow as tf
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class MeanLayer(OpticalLayer):
    def __init__(self, size = (100,100),interpolation = "nearest", **kwargs):
        super(MeanLayer,self).__init__(**kwargs)
        self.size = [int(size[0]/2), int(size[1]/2)]

    
    def call(self, input):
        middlex, middley = int(input.shape[2]/2), int(input.shape[1]/2)
        #lowx, highx =  middlex-self.size[0], middlex+self.size[0]
        #lowy, high y = middley-self.size[1], middley+self.size[1]
        input = input[:,middlex-self.size[0]:middlex+self.size[0], middley-self.size[1]:middley+self.size[1], :]
        res = tf.reduce_mean(input, axis = (1,2))

        return res

    def compute_output_shape(self, input_shape):
        if input_shape[3] == 1:
            return [input_shape[0], 1,input_shape[3]]
        else:
            return [input_shape[0],1,input_shape[3]]