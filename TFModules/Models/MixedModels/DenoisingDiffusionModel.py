import tensorflow as tf
from PhaseplateNetwork.TFModules.Models.DDNNModels.DiffractionPlateNetworkPooling import DiffractionPlateNetworkPooling
import PhaseplateNetwork.TFModules.NetworkLayers as NL

def beta_schedule(timesteps, beta_start = 0.001, beta_end = 0.01):
    return tf.linspace(beta_start, beta_end, timesteps)

class DenoisingDiffusionModel(tf.keras.Model):

    def __init__(self, input_dim = 30, optical_model = None, timesteps = 100, beta_start = 0.001, beta_end = 0.01, **kwargs):
        super(DenoisingDiffusionModel, self).__init__(**kwargs)
        if optical_model == None:
            self.optical_model = DiffractionPlateNetworkPooling(pooling_factor = 4,num_layers = 10,distance = 0.002,trainable_pixel=120, propagation_pixel = 120,
                                plate_scale_factor = 1, propagation_size = 0.002, padding = 0.002)
        else:
            self.optical_model = optical_model

        self.norm_layer = NL.NormalizingLayer()
        self.timesteps = timesteps
        self.input_dim = input_dim
        betas = beta_schedule(timesteps = timesteps, beta_start = beta_start, beta_end =beta_end)
        alphas = 1- betas
        self.alphas_bar = tf.math.cumprod(alphas, axis = 0)
        self.emb = NL.TimeEmbedding(img_dim = input_dim, max_time = timesteps)


    def call(self, input):

        inp, t = input
        #print(input[0].shape)
        #print(input[1].shape)
        t_emb = self.emb(tf.cast(t,dtype=tf.float32))
        mod_inp = tf.complex(inp+0.0000000001,0.0)*tf.math.exp(1j*tf.complex(t_emb,0.0))
        optical_out = tf.math.abs(self.optical_model(mod_inp))
        normalized_out = self.norm_layer(optical_out)
        return normalized_out    

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_input_shape(self):
        return ([None, self.input_dim, self.input_dim, 1],[None, 1])

    def get_image_variables(self):
        res = []
        for i in range(0,len(self.optical_model.elements)):
            el = self.optical_model.elements[i].get_image_variables()
            if el != None:
                res.append(el)
        res = tf.concat(res, axis = 0)
        #print(res)
        return res

    