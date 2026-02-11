import tensorflow as tf
from PhaseplateNetwork.TFModules.Models.DDNNModels.DiffractionPlateNetworkPoolingNormalized import DiffractionPlateNetworkPoolingNormalized
import PhaseplateNetwork.TFModules.NetworkLayers as NL


class OpticalGanModel(tf.keras.Model):
    def __init__(self, input_dim = 30, optical_model = None, timesteps = 100, beta_start = 0.001, beta_end = 0.01, **kwargs):
        super(OpticalGanModel, self).__init__(**kwargs)
        if optical_model == None:
            self.optical_model = DiffractionPlateNetworkPoolingNormalized(pooling_factor = 4,num_layers = 8,distance = 0.03,trainable_pixel=120, propagation_pixel = 120,
                                plate_scale_factor = 1, propagation_size = 0.002, padding = 0.002)
        else
            self.optical_model = optical_model

            

        
        


    def call(self, input):


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

    