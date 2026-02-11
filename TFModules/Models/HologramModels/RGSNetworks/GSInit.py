import PhaseplateNetwork.TFModules.Models.HologramModels.ResidualGerchbergSaxtonNetwork as ResidualGerchbergSaxtonNetwork
import numpy as np
import tensorflow as tf

class GSInit(ResidualGerchbergSaxtonNetwork):
    def __init__(self, z=np.array([0.01, 0.02, 0.03, 0.04]), network_depth=10, L=0.05, p=0.05, N=60,
                 transducer_radius=0.025, f0=1e6, cM=1484, print_info=False, **kwargs):
        super(GSInit,self).__init__(z , network_depth, L, p, N, transducer_radius, f0, cM, print_info,**kwargs)

    def get_hologram_and_output(self,Input):
        u = tf.complex(tf.zeros_like(Input[:,:,:,0:1]),tf.zeros_like(Input[:,:,:,0:1]))

        #init with one GB step
        inp = self.transducer_projection(u)
        prop = self.forward_propagation(inp)
        back = self.backward_propagation( self.image_projection(prop, Input))
        inp = self.transducer_projection(tf.math.reduce_mean(back, axis = 3, keepdims = True))


        for i in range(0,self.network_depth):
            propagated_images = self.forward_propagation(inp)
            stacked_goal_amplitudes = tf.concat((self.complex_to_channel(propagated_images), Input), axis = 3)
            conv_block_forward = self.forward_blocks[i](stacked_goal_amplitudes)
            propagated_holograms = self.backward_propagation( self.channel_to_complex(conv_block_forward))

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
        propagated_images = self.forward_propagation(inp)
        return propagated_images, inp
