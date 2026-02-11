import tensorflow as tf
import numpy as np
#from PhaseplateNetwork.TFModules.NetworkLayers.TransducerConstraintProjection import TransducerConstraintProjection
import PhaseplateNetwork.TFModules.NetworkLayers as NL
import PhaseplateNetwork.TFModules.OpticalLayers as OL
from PhaseplateNetwork.utils.conversion_utils import get_numpy_array_from_str

class ResidualGerchbergSaxtonNetwork(tf.keras.models.Model):
    def __init__(self, planes_distance = '0.1,0.2,0.3,0.4', network_depth = 10, propagation_size= 0.05, padding = 0.05, trainable_pixel = 60, transducer_radius= 0.025, frequency = 1e6, wavespeed = 1484,print_info = False,grad_normalization = False,normalization = 0.001, complex_mode = 'cartesian', **kwargs):
        super(ResidualGerchbergSaxtonNetwork, self).__init__(**kwargs)
        #self.transducer_projection = NL.TransducerConstraintProjection(L, N, transducer_radius)
        L = propagation_size
        p = padding
        N = trainable_pixel
        f0 = frequency
        cM = wavespeed
        self.transducer_projection = NL.TransducerConstraintProjection(L,N,transducer_radius = transducer_radius)
        self.image_projection = NL.AmplitudeConstraintProjection()
        self.network_depth = network_depth
        self.forward_blocks = []
        self.backward_blocks = []
        self.complex_to_channel = NL.ComplexToChannel(mode = complex_mode)
        self.channel_to_complex = NL.ChannelToComplex(mode = complex_mode)

        z = get_numpy_array_from_str(planes_distance)
        self.forward_propagation = OL.WavePropagation(z, N, L,p, f0, cM)
        self.super_gaussian = OL.MultiplyWithSuperGaussian(shape = [N,N])
        self.backward_propagation = OL.WavePropagation(-z, N, L, p, f0, cM)
        self.grad_normalization = grad_normalization
        self.normalization_layer = NL.GradNormalizationLayer(normalization)
        num_images = len(z)
        for i in range(0,network_depth):
            self.forward_blocks.append(NL.ConvBlock(out = 2*num_images))
            self.backward_blocks.append(NL.ConvBlock(out = 1))

        self.print_info = print_info



    def get_hologram_and_output(self,Input):
        u = tf.complex(tf.zeros_like(Input[:,:,:,0:1]),tf.zeros_like(Input[:,:,:,0:1]))
        inp = self.transducer_projection(u)
        for i in range(0,self.network_depth):
            propagated_images = self.forward_propagation(inp)
            propagated_images = self.super_gaussian(propagated_images)
            stacked_goal_amplitudes = tf.concat((self.complex_to_channel(propagated_images), Input), axis = 3)
            conv_block_forward = self.forward_blocks[i](stacked_goal_amplitudes)
            propagated_holograms = self.backward_propagation( self.channel_to_complex(conv_block_forward))
            propagated_holograms = self.super_gaussian(propagated_holograms)
            input_backward_blocks = tf.concat((self.complex_to_channel(propagated_holograms), tf.math.angle(inp)), axis = 3)
            hologram = self.backward_blocks[i](input_backward_blocks)
            inp = self.transducer_projection(hologram)

            if self.print_info:
                print('propagated_images_shape:{}'.format(propagated_images.shape))
                print('stacked_goal_amplitudes shape: {}'.format(stacked_goal_amplitudes.shape))
                print('conv_block_forward shape: {}'.format(conv_block_forward.shape))
                print('propagated_holograms shape: {}'.format(propagated_holograms.shape))
                print('hologram_shape: {}'.format(hologram.shape))
                print('hologram_dtype: {}'.format(hologram.dtype))
        if self.grad_normalization:
            #grad_loss = self.grad_loss(inp)
            inp = self.normalization_layer(inp)
            #self.add_loss(self.grad_loss(inp))

        propagated_images = self.forward_propagation(inp)*0.1
        return propagated_images, inp

    def call(self, Input, training = None, mask = None):
        images, hologram = self.get_hologram_and_output(Input)
        return images



    def get_hologram_plate(self, Input):
        images, hologram = self.get_hologram_and_output(Input)
        return hologram

    def get_image_variables(self, Input):
        hologram = self.get_hologram_plate(Input)
        return [hologram[0,:,:,0]]



