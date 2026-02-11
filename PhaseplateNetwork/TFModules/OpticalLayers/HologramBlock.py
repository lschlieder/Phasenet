import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
from PhaseplateNetwork.TFModules.OpticalLayers.WavePropagation import WavePropagation
from PhaseplateNetwork.TFModules.OpticalLayers.PhasePlate import PhasePlate
from PhaseplateNetwork.utils.util import repeat_image_tensor

EPSILON = 1e-16
class HologramBlock(OpticalLayer):
    def __init__(self, z, N, L, trainable_pixels,plate_scale, padding=None, num_plates = 10, f0=2.00e6, cM=1484, channels_last=True, use_FT=True, **kwargs):
        super(HologramBlock,self).__init__()
        z_temp = 0.0
        self.elements = []
        delta_z = z/num_plates
        #while z_temp < z:
        for plate in range(0,num_plates):
            self.elements.append(WavePropagation(delta_z, N , L , padding, f0, cM, channels_last, use_FT))
            self.elements.append(PhasePlate((trainable_pixels,trainable_pixels),plate_scale, False, True))
            #z_temp = z_temp + delta_z

        self.elements.append(WavePropagation(z - z_temp, N, L , padding , f0, cM, channels_last, use_FT))

    def call(self, input):
        inp = input
        for layer in self.elements:
            inp = layer(inp)
        return inp

    def get_image_variables(self):
        res = []
        for i in range(0,len(self.elements)):
            el = self.elements[i].get_image_variables()
            if el != None:
                res.append(el)
        res = tf.concat(res, axis = 0)
        #print(res)
        return res

    def compute_output_shape(self, input_shape):
        #assert( input_shape[1] == self.phases.shape[1])
        #assert( input_shape[2] == self.phases.shape[2])
        #assert( input_shape[3] == self.phases.shape[3])
        output_shape = []
        for i in range(0,len(input_shape)):
            output_shape.append(input_shape[i])

        return output_shape

    def call_for_fields(self, input, num = 100):
        fields = [input]
        out_fields = [input]
        running_input = input
        for layer in self.elements:
            running_input = layer(running_input)
            out_fields = layer.call_for_fields(running_input, num = num//len(self.elements))
            fields = fields + out_fields
        return fields

