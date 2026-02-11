import tensorflow as tf

epsilon = 1e-10

class ChannelToComplex(tf.keras.layers.Layer):
    def __init__(self, mode = 'cartesian', **kwargs):
        super(ChannelToComplex, self).__init__(**kwargs)
        self.mode = mode
        if mode not in ['cartesian', 'radial']:
            raise ValueError("self.mode had must be in [radial, cartesian]")

    def call(self, input):
        if input.shape[3] %2 != 0:
            raise ValueError("input.shape[channel_dim] must be divisible by 2. was {}".format(input.shape[self.channel_dim]))
        #slice_point = input.shape[self.channel_dim] // 2
        #slice_arr_real1 = [ 0, 0, 0]
        #slice_arr_real2 = [ input.shape[1], input.shape[2], input.shape[3]]
        #slice_arr_real2[self.channel_dim] = slice_point
        #slice_arr_imag1 = [ 0, 0, 0]
        #slice_arr_imag1[self.channel_dim-1] = slice_point
        #slice_arr_imag2 = [input.shape[1], input.shape[2], input.shape[3]]
        #slice_arr_imag2[self.channel_dim] = slice_point
        # print(slice_arr_imag1)
        # print(slice_arr_imag2)
        #a = tf.slice(input[:], slice_arr_real1, slice_arr_real2)
        #b = tf.slice(input[:], slice_arr_imag1, slice_arr_imag2)
        a = input[:,:,:,:input.shape[3]//2]
        b = input[:,:,:,input.shape[3]//2:]

        if self.mode == 'cartesian':
            ret = tf.complex(a, b)
        elif self.mode == 'radial':
            ret = tf.complex(a + epsilon, 0.0) * tf.math.exp(tf.complex(0.0, b))
        else:
            raise ValueError("self.mode had must be in [radial, cartesian]")
        return ret