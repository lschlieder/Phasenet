import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.OpticalLayers as OL
#import PhaseplateNetwork.TFModules.Models.DDNNModels as Models
#import PhaseplateNetwork.TFModules.Models.DDNNModels.JonesNetworkWithoutInput
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN
import importlib
import matplotlib.pyplot as plt


class RecurrentDDNN(DDNN):

    def __init__(self, NetworkModelForward = "JonesNetworkWithoutInput", NetworkModelBackward = 'JonesNetworkWithoutInput', num_iter = 4, trainable_pixel = 112, plate_scale_factor = 2, propagation_size = 0.001792, propagation_pixel = 224, padding = 0.00175, frequency = 3.843e14, wavespeed = 3e8,**kwargs):
        super(RecurrentDDNN,self).__init__(trainable_pixel=trainable_pixel, plate_scale_factor= plate_scale_factor, propagation_size = propagation_size, propagation_pixel = propagation_pixel, padding = padding, frequency = frequency, wavespeed =wavespeed,**kwargs)
        def class_for_name(module_name, class_name):
            # load the module, will raise ImportError if module cannot be loaded
            m = importlib.import_module(module_name)
            # get the class, will raise AttributeError if class cannot be found
            c = getattr(m, class_name)
            return c
        print('Num iterations of RDDNN: {}'.format(num_iter))
        #self.NetworkModel = NetworkModel
        self.NetworkModelForward = NetworkModelForward
        self.NetworkModelBackward = NetworkModelBackward
        if NetworkModelForward != None:
            self.forward_diffraction = class_for_name("PhaseplateNetwork.TFModules.Models.DDNNModels."+NetworkModel, NetworkModel)(trainable_polarization = 'x', trainable_pixel=trainable_pixel, plate_scale_factor= plate_scale_factor, propagation_size = propagation_size, propagation_pixel = propagation_pixel, padding = padding, frequency = frequency, wavespeed =wavespeed,**kwargs)
            self.elements.append(self.forward_diffraction)
        #else:
        #    self.forward_diffraction = None
        self.num_iter = num_iter
        if NetworkModelBackward != None:
            self.backward_diffraction = class_for_name("PhaseplateNetwork.TFModules.Models.DDNNModels."+NetworkModel, NetworkModel)(trainable_polarization = 'y', trainable_pixel = trainable_pixel, plate_scale_factor = plate_scale_factor, propagation_size = propagation_size, propagation_pixel = propagation_pixel, padding = padding, frequency = frequency, wavespeed =wavespeed,**kwargs)
            self.elements.append(self.backward_diffraction)

        self.Input_layer = OL.AmplitudeToIntensityJonesPolarization(axis  ='x')
        self.back_layer = OL.AmplitudeToIntensityJonesPolarization(axis = 'y')

    def append_networks(self):
        self.elements.append(self.forward_diffraction)
        self.elements.append(self.backward_diffraction)

    def call(self,Input):
        inp = self.Input_layer(Input)
        #inp = tf.concat((inp[:,:,:,0:1], tf.zeros_like(inp[:,:,:,0:1])),axis = 3)
        #inp = Input
        for i in range(0, self.num_iter):
            output = self.forward_diffraction(inp)

            back_inp = self.back_layer(output[:,:,:,1:2])
            #plt.imshow(np.abs(back_inp[0,:,:,0]))
            #plt.figure()
            #plt.imshow(np.abs(back_inp[0,:,:,1]))
            #plt.figure()
            #plt.imshow(np.abs(output[0,:,:,1]))
            #plt.figure()
            #plt.imshow(np.abs(output[0,:,:,0]))
            #plt.show()
            re = self.backward_diffraction(back_inp)

            inp = tf.concat((inp[:,:,:,0:1], re[:,:,:,1:2]), axis = 3)
        return output[:,:,:,0:1]


    def compute_output_shape(self, input_shape):
        return [None, input_shape[1], input_shape[2],1]

    def get_propagation_fields(self, Input):
        inp = self.Input_layer(Input)
        for i in range(0, self.num_iter):
            print(i)
            output = self.forward_diffraction(inp)
            back_inp = self.back_layer(output[:, :, :, 1:2])
            re = self.backward_diffraction(back_inp)
            inp = tf.concat((inp[:,:,:,0:1], re[:,:,:,1:2]), axis = 3)
        #output = self.forward_diffraction(inp)
        #re = self.backward_diffraction(output[:,:,:,0:1])

        forward = self.forward_diffraction.get_propagation_fields(inp)
        output = self.forward_diffraction(inp)
        back_inp = self.back_layer(output[:,:,:,1:2])
        backward = self.backward_diffraction.get_propagation_fields(back_inp)
        #self.save_fields([inp]+forward+backward+[output[:,:,:1:2]], PATH, False)
        return [inp.numpy()]+forward+backward+[output[:,:,:,0:1].numpy()]

    @classmethod
    def from_config(cls, config):
        RDDNN = cls(NetworkModel = config['NetworkModel'], num_iter= config['num_iter'],trainable_pixel=config['trainable_pixel'], scaling=config['scaling'], L = config['L'], propagation_pixel=config['propagation_pixel'], padding = config['padding'], f0 = config['f0'], cM = config['cM'])

        input_shape =  [None, config['propagation_pixel'], config['propagation_pixel'], 1]

        for element in config['elements']:
            element_class = globals()[element['class_name']]
            #print(input_shape)
            print('Created {} with input shape {}'.format(element['class_name'], input_shape))
            input_shape = RDDNN.append_element(element_class.from_config(element['config']), input_shape)

        self.forward_diffraction = self.elements[0]
        self.backward_diffraction = self.elements[1]
        return RDDNN

    def get_image_variables(self):
        res = []
        for i in range(0,len(self.elements)):

            el = self.elements[i].get_image_variables()
            if el != None:

                res.append(el)
            #print(i)
            #print(el.shape)


        #print(self.trainable_variables)
        res = tf.concat(res, axis = 0)
        #print(res.shape)
        return res

    def get_config(self):
        #elements_dict = {}
        elements_arr = []
        input_shape = [None,self.propagation_pixel, self.propagation_pixel, 1]
        for l in self.elements:
            print(l)
            #elements_dict[l.name] = {'class_name': l.__class__.__name__, 'config':l.get_config()}
            elements_arr.append({'name': l.name, 'class_name': l.__class__.__name__, 'input_shape':input_shape, 'config':l.get_config()})
            input_shape = l.compute_output_shape(input_shape)
        temp = {
            'trainable_pixel': self.trainable_pixel,
            'scaling': self.scaling,
            'L':self.L,
            'propagation_pixel': self.propagation_pixel,
            'padding': self.padding,
            'f0': self.f0,
            'cM': self.cM,
            'num_iter': self.num_iter,
            'NetworkModelForward': self.NetworkModelForward,
            'NetworkModelBackward': self.NetworkModelBackward,
            'elements': elements_arr
        }
        return temp



