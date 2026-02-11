import tensorflow as tf
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class ReduceDimension(OpticalLayer):
    def __init__(self, M,N, image_axis = 0):
        super(ReduceDimension,self).__init__()
        self.M = M
        self.N = N

        self.image_axis = image_axis

    def call(self,input):
        #print(input.shape)
        assert self.M*self.N == input.shape[2 - self.image_axis]
        temp = tf.math.reduce_mean( input, axis = self.image_axis + 1)
        out = tf.reshape(temp, (temp.shape[0], self.M,self.N, input.shape[3]))
        return out

    def compute_output_shape(self, input_shape):
        print(input_shape)
        print(self.M)

        print(self.N)
        print(self.M*self.N)
        print(input_shape[2-self.image_axis])
        assert self.M*self.N == input_shape[2 - self.image_axis]
        output_shape = []
        for i in range(0,len(input_shape)):
            if len(input_shape) - i == 3:
                output_shape.append(self.M)
            elif len(input_shape)-i == 2:
                output_shape.append(self.N)
                #output_shape.append(input_shape[1] * input_shape[2])
            else:
                output_shape.append(input_shape[i])
        return output_shape

