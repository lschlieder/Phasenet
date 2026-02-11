import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.NetworkLayers as NL
from PhaseplateNetwork.TFModules.Models.DDNNModels.DiffractionPlateNetwork import DiffractionPlateNetwork
import inspect
    

    
class MultiNetworkModel(tf.keras.models.Model):
    #def __init__(self,num_passes = 4, num_layers = 4, phase_input = True, distance = 0.04, trainable_pixel = 224, plate_scale_factor = 1, propagation_size = 0.001792, propagation_pixel = 224, padding = 0.00175, frequency = 3.843e14, wavespeed = 3e8, **kwargs):
    def __init__(self, num_passes = 4, weave_num = 4, unweave_num = 4, input_size = (28,28), propagation_pixel = 112, **kwargs):   
        signature = inspect.signature(DiffractionPlateNetwork.__init__)
        model_args = {}
        super_args = {}
        print(kwargs)
        for k in kwargs.keys():
            if k in signature.parameters:
                model_args[k] = kwargs[k]
            else:
                super_args[k] = kwargs[k]
        
        print(super_args)
        print(model_args)
        super(MultiNetworkModel,self).__init__(**super_args)
        self.models = []
        self.propagation_pixel = propagation_pixel
        for i in range(0,num_passes):
            #m = DiffractionPlateNetwork(num_layers = 4, phase_input = True, distance = 0.04, trainable_pixel = 112, plate_scale_factor = 2, propagation_size = 0.001792, propagation_pixel = 224, padding = 0.00175, frequency = 3.843e14, wavespeed = 3e8)
            m = DiffractionPlateNetwork(propagation_pixel = propagation_pixel, **model_args)
            self.models.append(m )
            
        self.weave = NL.WeaveLayer(weave_num)
        self.unweave = NL.UnweaveLayer(unweave_num, (self.propagation_pixel,self.propagation_pixel))
        
        self.t_encoding = tf.keras.layers.Dense(input_size[0]*input_size[1])
        self.input_size = input_size
        self.offset= tf.Variable(1.0,trainable=True)
            
        #self.mask =  get_chess_mask((8,8), (trainable_pixel, trainable_pixel) )
        
    def call(self, input):
        #u = input
        t = input[1]
        
        t_enc = tf.reshape( self.t_encoding(t), (-1, self.input_size[0],self.input_size[1],1))
        
        print(input[0].shape)
        print(input[1].shape)
        inp_res = tf.image.resize(input[0], size = (self.propagation_pixel//2,self.propagation_pixel//2))
        t_enc_res = tf.image.resize(t_enc, size = (self.propagation_pixel//2,self.propagation_pixel//2))

        z = tf.zeros_like(inp_res)
        #inp_imgs = self.weave(tf.concat(( inp_res, z, z, t_enc_res), axis = 3))


        #u = tf.image.resize(inp_imgs, (self.propagation_pixel, self.propagation_pixel))
        #u = inp_imgs
        u = tf.zeros_like(inp_res)

        for m in self.models:
            u = self.weave(tf.concat( (inp_res, u, u, t_enc_res), axis = 3))
            u = m(u)
            u = self.unweave(u)
            u = u[:,:,:,0:1] - u[:,:,:,2:3]

            #print(u.shape)
        
        #u = self.unweave(u)
        
        #u = u[:,:,:,0:1] - u[:,:,:,2:3]
        u = u*self.offset
        u = tf.image.resize(u, (28,28))
        return u 