import tensorflow as tf

class Flatten(tf.keras.layers.Layer):
    def __init__(self, image_axis = 0):
        '''
        Flattens an image to a single dimension and tiles it. an image of input size NxM would have size N*M x N*M afterwards
        '''
        super(Flatten,self).__init__()
        self.image_axis = image_axis



    def call(self,input):
        if self.image_axis == 0:
            N = input.shape[1]*input.shape[2]
            temp = tf.reshape( input, (input.shape[0], 1, N, input.shape[3]))
            output = tf.tile( temp, (1,N, 1,1 ))
        elif self.image_axis == 1:
            N = input.shape[1]*input.shape[2]
            temp = tf.reshape( input, (input.shape[0], N, 1, input.shape[3]))
            output = tf.tile( temp, (1,1, N,1 ))
        return output

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(0,len(input_shape)):

            if len(input_shape) - i == 3 or len(input_shape)-i == 2:
                #if self.image_axis == 0:
                output_shape.append(input_shape[1] * input_shape[2])
            else:
                output_shape.append(input_shape[i])

        return output_shape
