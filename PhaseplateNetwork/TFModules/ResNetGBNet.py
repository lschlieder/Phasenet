import tensorflow as tf
from PhaseplateNetwork.TFModules.NetworkLayers.ConvBlock import ConvBlock
import numpy as np

epsilon = 1e-10


class ResNetGBNet(tf.keras.Model):

    def __init__(self, forward_operator, image_constraint_projection, depth = 10, layer_information = [ConvBlock, 'linear'], res = True):
        super(ResNetGBNet,self).__init__()

        self.conv_blocks_forward = []
        self.conv_blocks_backward = []

        self.forward_operator = forward_operator
        #self.backward_operator = backward_operator
        self.image_constraint_projection = image_constraint_projection

        layer_type, layer_activation = layer_information
        self.network_depth = depth
        self.layer_type = layer_type
        self.layer_activation = layer_activation

        out_shape = forward_operator.compute_output_shape([10, 60,60,1])[3]

        self.use_residual_connection = res

        for i in range(0, depth):
            self.conv_blocks_forward.append(layer_type(out_shape * 2))
            self.conv_blocks_backward.append(layer_type(1, out_activation=layer_activation))

    @tf.function
    def call(self, inputs, **kwargs):
        img = inputs

        u = tf.constant(np.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2], 1)))
        u = self.image_constraint_projection(u)
        u_array = [u]

        for i in range(0, self.network_depth):
                #Calculate the forward operator output
                proj = complex_to_channel(self.forward_operator(u))
                proj = tf.concat((proj, img), axis = 3)
                #Calculate the learned block output
                image_plane = channel_to_complex(self.conv_blocks_forward[i].call(proj))
                #Calculate the backward propagated output
                back_proj = complex_to_channel(self.backward_operator(image_plane))
                back_proj = tf.concat((back_proj, tf.math.angle(u)), axis = 3)
                hologram_plane = np.pi*( self.conv_blocks_backward[i].call(back_proj))
                if self.use_residual_connection:
                    u = u + self.image_constraint_projection(hologram_plane)
                    #u = u + (self.amplitude + epsilon) * tf.exp(1j * tf.cast(hologram_plane, dtype=tf.complex64))
                else:
                    u = self.image_constraint_projection(hologram_plane)
                    #u = (self.amplitude + epsilon) * tf.exp(1j * tf.cast(hologram_plane, dtype=tf.complex64))

                u_array.append(u)

        res = self.forward_operator(u)

        return res, u_array



@tf.function
def complex_to_channel(input, channel_dim=3):
    real = tf.math.real(input)
    imag = tf.math.imag(input)
    return tf.concat([real, imag], channel_dim)


@tf.function
def complex_to_channel_radial(input, channel_dim=3):
    abs = tf.math.abs(input)
    angle = tf.math.angle(input)
    return tf.concat([abs, angle], channel_dim)


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


# @tf.function
def channel_to_complex_radial(input, channel_dim=3):
    slice_arr_abs1 = [0, 0, 0, 0]
    slice_arr_abs2 = [input.shape[0], input.shape[1], input.shape[2], input.shape[3]]
    slice_arr_abs2[channel_dim] = input.shape[channel_dim] // 2
    slice_arr_angle1 = [0, 0, 0, 0]
    slice_arr_angle1[channel_dim] = input.shape[channel_dim] // 2
    slice_arr_angle2 = [input.shape[0], input.shape[1], input.shape[2], input.shape[3]]
    slice_arr_angle2[channel_dim] = input.shape[channel_dim] // 2
    abs_img = tf.slice(input, slice_arr_abs1, slice_arr_abs2)
    angle = tf.slice(input, slice_arr_angle1, slice_arr_angle2)
    ret = tf.complex(abs_img + epsilon, 0.0) * tf.math.exp(tf.complex(0.0, angle))

    return ret



