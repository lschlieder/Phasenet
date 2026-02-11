import tensorflow as tf

class DeepCGHNetwork(tf.keras.Model):

    def __init__(self, forward_operator, image_constraint_projection, network_block, res = True):
        super(DeepCGHNetwork,self).__init__()

        self.conv_blocks_forward = []
        self.conv_blocks_backward = []

        self.forward_operator = forward_operator
        #self.backward_operator = backward_operator
        self.image_constraint_projection = image_constraint_projection

        #layer_type, layer_activation = layer_information
        #self.layer_type = layer_type
        #self.layer_activation = layer_activation

        out_shape = forward_operator.compute_output_shape([10, 60,60,1])[3]

        self.network_block = network_block

    def call(self,input):
        #Hologram_complex = channel_to_complex(self.network_block(input))
        Hologram_complex = self.network_block(input)
        print(Hologram_complex.shape)
        print(Hologram_complex.dtype)
        Hologram = self.image_constraint_projection(self.forward_operator.inverse_call(Hologram_complex))

        res = self.forward_operator(Hologram)
        return res, Hologram



@tf.function
def complex_to_channel(input, channel_dim=3):
    real = tf.math.real(input)
    imag = tf.math.imag(input)
    return tf.concat([real, imag], channel_dim)

@tf.function
def channel_to_complex(input, channel_dim=3):
    slice_arr_real1 = [0, 0, 0, 0]
    slice_arr_real2 = [input.shape[0], input.shape[1], input.shape[2], input.shape[3]]
    slice_arr_real2[channel_dim] = input.shape[channel_dim] // 2
    slice_arr_imag1 = [0, 0, 0, 0]
    slice_arr_imag1[channel_dim] = input.shape[channel_dim] // 2
    slice_arr_imag2 = [input.shape[0], input.shape[1], input.shape[2], input.shape[3]]
    slice_arr_imag2[channel_dim] = input.shape[channel_dim] // 2
    # print(slice_arr_imag1)
    # print(slice_arr_imag2)
    real = tf.slice(input, slice_arr_real1, slice_arr_real2)
    imag = tf.slice(input, slice_arr_imag1, slice_arr_imag2)
    ret = tf.complex(real, imag)
    return ret