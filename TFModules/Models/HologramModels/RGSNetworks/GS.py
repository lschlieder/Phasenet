import PhaseplateNetwork.TFModules.Models.HologramModels.ResidualGerchbergSaxtonNetwork as ResidualGerchbergSaxtonNetwork
import numpy as np
import tensorflow as tf

class GS(ResidualGerchbergSaxtonNetwork):
    def __init__(self, z=np.array([0.01, 0.02, 0.03, 0.04]), network_depth=10, L=0.05, p=0.05, N=60,
                 transducer_radius=0.025, f0=1e6, cM=1484, print_info=False, **kwargs):
        super(GS,self).__init__(z , network_depth, L, p, N, transducer_radius, f0, cM, print_info,**kwargs)

    def get_hologram_and_output(self,Input):
        u = tf.complex(tf.zeros_like(Input[:,:,:,0:1]),tf.zeros_like(Input[:,:,:,0:1]))

        #init with one GB step
        inp = self.transducer_projection(u)
        prop = self.forward_propagation(inp)
        back = self.backward_propagation( self.image_projection(prop, Input))
        inp = self.transducer_projection(tf.math.reduce_mean(back, axis = 3, keepdims = True))


        for i in range(0,self.network_depth):
            prop = self.forward_propagation(inp)
            back = self.backward_propagation(self.image_projection(prop, Input))
            inp = self.transducer_projection(tf.math.reduce_mean(back, axis=3, keepdims=True))
        propagated_images = self.forward_propagation(inp)
        return propagated_images, inp
