import tensorflow as tf
#from PhaseplateNetwork.TFModules.NetworkLayers.TransducerConstraintProjection import TransducerConstraintProjection
import PhaseplateNetwork.TFModules.NetworkLayers as NL
import PhaseplateNetwork.TFModules.OpticalLayers as OL
class ResidualGerchbergSaxtonNetwork(tf.keras.models.Model):
    def __init__(self, z = np.array([0.1,0.2,0.3,0.4]), network_depth = 10, L = 0.05, p = 0.05, N = 60, transducer_radius= 0.025, f0 = 1e6, cM = 1484, **kwargs):
        super(ResidualGerchbergSaxtonNetwork, self).__init__(**kwargs)
        #self.transducer_projection = NL.TransducerConstraintProjection(L, N, transducer_radius)
        self.transducer_projection = NL.AmplitudeConstraintProjection()
        self.image_projection = NL.AmplitudeConstraintProjection()
        self.network_depth = network_depth
        self.forward_blocks = []
        self.backward_blocks = []
        self.complex_to_channel = NL.ComplexToChannel(mode = 'cartesian')
        self.channel_to_complex = NL.ChannelToComplex(mode = 'cartesian')
        self.forward_propagation = OL.WavePropagation(z, N, L,p, f0, cM)
        self.backward_propagation = OL.WavePropagation(-z, N, L, p, f0, cM)
        for i in range(0,network_depth):
            self.forward_blocks.append(NL.ConvBlock(out = 2))
            self.backward_blocks.append(NL.ConvBlock(out = 2))

    def call(self, Input, training = None, mask = None):
        #transducer_constraint = Input[:,0,:,:,:]
        #image_constraint = Input[:,1,:,:,:]

        u = tf.complex(tf.zeros_like(transducer_constraint),tf.zeros_like(transducer_constraint))
        inp = self.transducer_projection(u,transducer_constraint)








    def call(self, inputs, training=None, mask=None):
        for i in range(0,self.network_depth):


