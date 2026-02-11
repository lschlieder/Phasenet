import tensorflow as tf
import pickle as pkl
import PhaseplateNetwork.TFModules.OpticalLayers as OL
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalNonlinearity import OpticalNonlinearity
from PhaseplateNetwork.TFModules.OpticalLayers.SaturableAbsorber import SaturableAbsorber
from PhaseplateNetwork.TFModules.OpticalLayers.ReluNonlinearity import ReluNonlinearity
from PhaseplateNetwork.TFModules.OpticalLayers.PhasePlate import PhasePlate
from PhaseplateNetwork.TFModules.OpticalLayers.WavePropagation import WavePropagation
from PhaseplateNetwork.TFModules.OpticalLayers.ReduceDimension import ReduceDimension
from PhaseplateNetwork.TFModules.OpticalLayers.Flatten import Flatten
from PhaseplateNetwork.TFModules.OpticalLayers.AmplitudeAveragePooling2D import AmplitudeAveragePooling2D
from PhaseplateNetwork.TFModules.OpticalLayers.ZernikePhasePlate import ZernikePhasePlate
from PhaseplateNetwork.TFModules.OpticalLayers.TrainableLensePhasePlate import TrainableLensePhasePlate
from PhaseplateNetwork.TFModules.OpticalLayers.MultilenseArrayPhasePlate import MultilenseArrayPhasePlate
from PhaseplateNetwork.TFModules.OpticalLayers.LensePhasePlate import LensePhasePlate
from PhaseplateNetwork.TFModules.OpticalLayers.HologramBlock import HologramBlock
from PhaseplateNetwork.TFModules.OpticalLayers.PaddingLayer import PaddingLayer
from PhaseplateNetwork.TFModules.OpticalLayers.CropLayer import CropLayer
from PhaseplateNetwork.TFModules.OpticalLayers.AmplitudeToJonesPolarizationLayer import AmplitudeToJonesPolarizationLayer
from PhaseplateNetwork.TFModules.OpticalLayers.AbsFromJonesPolarizationLayer import AbsFromJonesPolarizationLayer
from PhaseplateNetwork.TFModules.OpticalLayers.AmplitudeToLeftRightPolarization import AmplitudeToLeftRightPolarization
from PhaseplateNetwork.TFModules.OpticalLayers.JonesPhasePlate import JonesPhasePlate
from PhaseplateNetwork.TFModules.OpticalLayers.LinearPolarizationFilter import LinearPolarizationFilter
from PhaseplateNetwork.TFModules.OpticalLayers.LCD_Display import LCD_Display
from PhaseplateNetwork.TFModules.OpticalLayers.AmplitudeToIntensityJonesPolarization import AmplitudeToIntensityJonesPolarization
from PhaseplateNetwork.TFModules.OpticalLayers.UncertaintyPhasePlate import UncertaintyPhasePlate
from PhaseplateNetwork.utils.plotting_utils import get_polarization_animation, save_gif_animation, get_field_animation
import numpy as np
import matplotlib.pyplot as plt


EPSILON = 1e-16
class DDNN(tf.keras.Model):

    def append_element(self, element, input_shape):
        
        #print(len(self.elements))
        if len(self.elements) == 0 :
            self.i_shape = input_shape
        self.elements.append(element)
        o_shape = element.compute_output_shape(input_shape)
        print('pos: {}, Appended {}, input_shape: {}, output_shape{}'.format(len(self.elements)-1,type(element).__name__, input_shape, o_shape))
        #print(o_shape)
        self.output_shapes.append(o_shape)
        self.o_shape = o_shape
        return o_shape

    def compute_output_shape(self, input_shape):
        return self.o_shape

    def clear_elements(self):
        self.elements = []
        self.output_shapes = []
        self.o_shape = self.get_input_shape()


    def get_layer_function(self, str):
        '''
        returns function for the creation of the layer defined in str
        str: line with an optical element and parameters
        '''
        element_dict= {
            'wave_propagation': 'self.get_wave_propagation(n_p = n_p,',
            'phase_plate': 'self.get_phaseplate(',
            'saturable_absorber': 'self.get_optical_nonlinearity_sat_abs(n_p=n_p,'
        }
        line = str.split(" ")

        element_fun = element_dict[line[0]]
        i = 0
        for args in line[1:]:
            if not i == 0:
                element_fun = element_fun+','
            element_fun= element_fun + args[:-1]
            i = i+1
        element_fun = element_fun+')'
        print(element_fun)
        #input()
        return lambda n_p: eval(element_fun)

    def read_network_configuration(self, network_config_file, input_shape):
        try:
            with open(network_config_file,'r') as f:
                config_strs = f.readlines()
        except:
            print("could not find network_config_file: {}".format(network_config_file))

        for line in config_strs:
            elem_fun = self.get_layer_function(line)
            print(elem_fun)
            print(self)
            input()
            input_shape = self.append_element(elem_fun(input_shape[1]), input_shape)
        return

    @classmethod
    def from_config(cls, config):
        DDNN = cls(trainable_pixel=config['trainable_pixel'], scaling=config['scaling'], L = config['L'], propagation_pixel=config['propagation_pixel'], padding = config['padding'], f0 = config['f0'], cM = config['cM'])

        input_shape =  [None, config['propagation_pixel'], config['propagation_pixel'], 1]
        for element in config['elements']:
            element_class = globals()[element['class_name']]
            #print(input_shape)
            print('Created {} with input shape {}'.format(element['class_name'], input_shape))
            input_shape = DDNN.append_element(element_class.from_config(element['config']), input_shape)

        return DDNN


    @classmethod
    def from_network_str(cls,network_str, trainable_pixel = 100, plate_scale_factor = 1, propagation_size = 0.08, propagation_pixel = 100, padding = 0.08, frequency = 2.00e6, wavespeed = 1484, ** kwargs):


        DDNN = cls(trainable_pixel, plate_scale_factor, propagation_size, propagation_pixel, padding, frequency, wavespeed)
        input_shape = [None, propagation_pixel, propagation_pixel, 1]
        DDNN.network_str = network_str
        DDNN.setup_network(network_str, input_shape)


        return DDNN


    def get_config(self):
        #elements_dict = {}
        elements_arr = []
        #input_shape = [None,self.propagation_pixel, self.propagation_pixel, 1]
        input_shape = self.get_input_shape()
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
            'elements': elements_arr
        }
        return temp


    def setup_network(self, network_str, input_shape):
        self.clear_elements()
        temp_num_str = ''
        nonlinearity = False
        hologram_block = False
        n_type = ''
        #print(input_shape)
        input_shape = self.append_element(self.get_padding_layer((input_shape[1],input_shape[2])), input_shape)
        for i in range(0, len(network_str)):
            char = network_str[i]
            #print(char)
            if char in ['0','1','2','3','4','5','6','7','8','9','.']:
                temp_num_str = temp_num_str + char
                if i != len(network_str)-1:
                    if network_str[i+1] not in ['0','1','2','3','4','5','6','7','8','9','.'] and network_str[i+1] in ['}', ']', ')']:
                        if nonlinearity:
                            if n_type == 'relu':
                                #self.elements.append( self.get_optical_nonlinearity(float(temp_num_str)))
                                input_shape = self.append_element(self.get_optical_nonlinearity(float(temp_num_str), input_shape[1]), input_shape)
                                temp_num_str = ''
                            elif n_type == 'sat_abs':
                                #self.elements.append(self.get_optical_nonlinearity_sat_abs( float(temp_num_str)))
                                input_shape = self.append_element(self.get_optical_nonlinearity_sat_abs( float(temp_num_str),input_shape[1]), input_shape)
                                temp_num_str = ''
                            elif n_type == 'amp':
                                input_shape = self.append_element(self.get_amplitude_nonlinearity(float(temp_num_str), input_shape[1]), input_shape)
                                temp_num_str = ''
                        else:
                            raise ValueError("Network String not correctly formated: {}".format(network_str))

                    elif network_str[i+1] not in ['0','1','2','3','4','5','6','7','8','9','.'] and network_str[i+1] in ['&']:
                        if hologram_block:
                            input_shape = self.append_element(self.get_hologram_block(float(temp_num_str), input_shape[1]), input_shape)
                            temp_num_str = ''
                        else:
                            input_shape = self.append_element(self.get_wave_propagation([float(temp_num_str)] ,input_shape[1]), input_shape)
                            #input_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])),input_shape)
                            #input_shape = self.append_element(self.get_padding_layer((input_shape[1], input_shape[2])),input_shape)
                            temp_num_str = ''

                    elif network_str[i+1] not in ['0','1','2','3','4','5','6','7','8','9','.'] and not network_str[i+1] in ['}',']', ')'] :
                        if not nonlinearity:
                            #self.elements.append( self.get_wave_propagation([float(temp_num_str)] ))
                            input_shape = self.append_element(self.get_wave_propagation([float(temp_num_str)] ,input_shape[1]), input_shape)
                            #input_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])),input_shape)
                            #input_shape = self.append_element(self.get_padding_layer((input_shape[1], input_shape[2])),input_shape)

                            temp_num_str = ''
                        else:
                            raise ValueError("Network String not correctly formated: {}".format(network_str))
                else:
                    if not nonlinearity:
                        #self.elements.append( self.get_wave_propagation([float(temp_num_str)]))
                        input_shape = self.append_element(self.get_wave_propagation([float(temp_num_str)],input_shape[1]), input_shape )
                        temp_num_str = ''
                    else:
                        raise Valueerror("Network String not correctly formated: {}".format(network_str))

            elif char == ',' and not nonlinearity:
                #self.elements.append( self.get_phaseplate(False, True))
                input_shape = self.append_element(self.get_phaseplate(False, True), input_shape)

            elif char == 'z' and not nonlinearity:
                input_shape = self.append_element(self.get_zernike_phaseplate(input_shape[1]), input_shape)

            elif char == 'l' and not nonlinearity:
                input_shape = self.append_element( self.get_lense_phaseplate(input_shape[1]), input_shape)
            #elif char == '.'and not nonlinearity:
                #self.elements.append( self.get_phaseplate(True, False))
            #    input_shape = self.append_element(self.get_phaseplate(True, False,input_shape[1]), input_shape)
            #   print('append phaseplate')
            elif char == 'f' and not nonlinearity:
                input_shape = self.append_element( self.get_lense_phaseplate(), input_shape)
            elif char == 'm' and not nonlinearity:
                input_shape = self.append_element(self.get_multiarray_lense_phaseplate(), input_shape)

            elif char == ';' and not nonlinearity:
                input_shape = self.append_element(self.get_phaseplate(True, True), input_shape)
            elif char == '-' and not nonlinearity:
                #self.elements.append( self.get_flatten())
                input_shape = self.append_element( self.get_flatten(), input_shape)
            elif char == '+' and not nonlinearity:
                #self.elements.append(self.get_reduce_dimension())
                input_shape = self.append_element( self.get_reduce_dimension(), input_shape)
            elif char == '#' and not nonlinearity:
                input_shape = self.append_element( self.get_amplitude_average_pooling(),input_shape)

            elif (char == '{' or char == '[' or char == '(') and not nonlinearity:
                nonlinearity = True
                if char== '{':
                    n_type = 'relu'
                elif char == '[':
                    n_type = 'sat_abs'
                elif char == '(':
                    n_type = 'amp'

            elif (char == '}' or char == ']' or char == ')') and nonlinearity:
                nonlinearity = False
                if char == '}' and not n_type == 'relu':
                    raise ValueError("Network String not correctly formated: {}".format(network_str))
                elif char ==']' and not n_type == 'sat_abs':
                    raise ValueError("Network String not correctly formated: {}".format(network_str))
                elif char ==')' and not n_type == 'amp':
                    raise ValueError("NetworkString not correctly formated: {}, {}".format(network_str, char))
            elif char == '&':
                if not hologram_block:
                    hologram_block = True
                else:
                    hologram_block = False



            else:
                raise ValueError("Network String not correctly formated: {}".format(network_str))



        input_shape = self.append_element(self.get_cropping_layer((input_shape[1], input_shape[2])), input_shape)
        return input_shape


    def get_input_shape(self):
        return self.i_shape#[None, self.elements[0].input_shape, self.elements[0].input_shape, 1]

    def get_propagation_size(self):
        return self.L
    

    def get_field_at_position(self, Input, pos):
        z = 0.0
        i = 0 
        u = Input
        while z < pos:
            layer = self.elements[i]
            if hasattr(layer, 'z'):
                print(layer.z)
                z = z + layer.z
            
            if z < pos:
                u = layer(u)
            else:
                prop_size = pos - (z - layer.z)
                last_prop_layer = self.get_wave_propagation([prop_size] ,self.output_shapes[i-1][1])
                u = last_prop_layer.call(u)

            i = i+1

        return u
    
    def get_propagation_distance(self):
        z = 0.0
        for layer in self.elements:
            if hasattr(layer, 'z'):
                print(layer.z)
                z = z + layer.z
        return z
    
    



    def __init__(self, trainable_pixel=100, plate_scale_factor = 1, propagation_size=0.08, propagation_pixel = 100, padding=0.08, frequency=2.00e6, wavespeed=1484, output_mode = False,**kwargs):
        super(DDNN,self).__init__(**kwargs)


        self.elements = []
        self.output_shapes = []

        self.trainable_pixel = trainable_pixel
        self.propagation_pixel = propagation_pixel
        #self.scale = scale
        self.propagation_size = propagation_size
        self.L = propagation_size
        L = self.L

        self.frequency = frequency
        f0 = frequency
        self.f0 =f0
        self.wavespeed = wavespeed
        self.cM = wavespeed
        cM = self.cM
        self.wl = self.cM/self.f0
        self.padding = padding
        self.plate_scale_factor = plate_scale_factor
        self.scaling = self.plate_scale_factor
        scaling = self.scaling

        self.wavelength = self.wavespeed/self.frequency

        #phaseplate_shape = [trainable_pixel,trainable_pixel]
        self.i_shape = None
        print(padding)
        self.get_wave_propagation = lambda z,n_p: WavePropagation( z, n_p, L+2*padding, padding, f0, cM, True)
        self.get_wave_propagation_without_padding = lambda z, n_p: WavePropagation(z, n_p, L, padding, f0, cM, True)
        # z, N, L, scale = 2, padding = None, f0 = 2.00e6, cM = 1484, channels_last = True, use_FT = True,
        self.get_phaseplate = lambda amp, phase: PhasePlate( [trainable_pixel,trainable_pixel], scaling, amp, phase)
        self.get_uncertainty_phaseplate = lambda amp, phase, sigma_phase, sigma_amp: UncertaintyPhasePlate([trainable_pixel, trainable_pixel], scaling, amp, phase, sigma_amp, sigma_phase)
        self.get_jones_phaseplate = lambda amp,phase, action: JonesPhasePlate( [trainable_pixel,trainable_pixel], scaling, amp, phase, action)
        self.get_rotating_plate = lambda : OL.PolarizationRotationPlate( [trainable_pixel,trainable_pixel], scaling)
        self.get_rotating_plate_constraint = lambda max_angle: OL.PolarizationRotationPlate( [trainable_pixel,trainable_pixel], scaling, max_angle)
        self.get_lcd_display = lambda alpha = (np.pi/2)/(5*10**-4), input_angle = 0.0, beta = 700000, depth = 5*10**-4: LCD_Display([trainable_pixel,trainable_pixel],scaling, True,alpha, input_angle,beta, depth)
        self.get_zernike_phaseplate = lambda n_p:ZernikePhasePlate( [n_p,n_p])
        self.get_trainable_lense_phaseplate = lambda n_p: TrainableLensePhasePlate( [n_p,n_p])
        self.get_lense_phaseplate = lambda : LensePhasePlate( [trainable_pixel, trainable_pixel], L,scaling, cM/f0)
        self.get_multiarray_lense_phaseplate = lambda : MultilenseArrayPhasePlate( [trainable_pixel, trainable_pixel], L, scaling, cM/f0, focus = 0.1, multi = 8)

        #activation_fn = lambda x: tf.cast(tf.nn.leaky_relu(tf.abs(x)+EPSILON-0.1), dtype = tf.complex64) * tf.math.exp(1j*tf.cast(tf.math.angle(x), dtype = tf.complex64))
        #self.get_optical_nonlinearity = lambda z,n_p: OpticalNonlinearity( z, 0.01, n_p, L, 1, activation_fn, padding, f0, cM, use_FT)
        self.get_optical_nonlinearity = lambda z, n_p: ReluNonlinearity(z, 0.001, n_p, L+2*padding,padding, f0, cM)

        amp_fun = lambda x: tf.cast(tf.abs(x), dtype = tf.complex64)
        self.get_amplitude_nonlinearity = lambda z,n_p: OpticalNonlinearity( z, 0.001, n_p, L+2*padding, amp_fun, padding, f0, cM)

        #z, N, L, trainable_pixels, scale = 1, padding = None, delta_z = 0.001, f0 = 2.00e6, cM = 1484, channels_last = True, use_FT = True, ** kwargs):
        self.get_hologram_block = lambda z, n_p, num_plates: HologramBlock(z, n_p, L+2*padding,trainable_pixel, scaling,padding,num_plates, f0, cM, True)

        self.get_optical_nonlinearity_sat_abs = lambda z,n_p: SaturableAbsorber(z, 0.001, n_p, L+2*padding, 0.0, f0, cM)

        self.get_flatten = lambda : Flatten()
        self.get_reduce_dimension = lambda: ReduceDimension(trainable_pixel, trainable_pixel)
        self.get_amplitude_average_pooling = lambda: AmplitudeAveragePooling2D((trainable_pixel,trainable_pixel))

        self.get_padding_layer = lambda n_p: PaddingLayer(n_p, L, padding )
        self.get_padding_layer_var = lambda n_p, v_padding: PaddingLayer(n_p,L, v_padding)
        self.get_cropping_layer = lambda n_p: CropLayer(n_p, L , padding)
        self.get_cropping_layer_var = lambda n_p, v_padding: CropLayer(n_p,L,v_padding)

        #self.read_network_configuration(network_file, input_shape)
        #self.average_pool = tf.keras.layers.AveragePooling2D(pool_size=(self.scale,self.scale))
        #self.phase_plates = tf.complex(self.amplitudes+EPSILON,0.0) * tf.cast( tf.exp( 1j * tf.cast(self.phases, dtype = tf.complex64)), dtype = tf.complex64)

    #@tf.function
    def call(self,Input, **kwargs):
        u = tf.cast(Input,dtype = tf.complex64)

        #u = repeat_image_tensor(u,self.scale)

        for layer in self.elements:
            #print(u.shape)
            u = layer.call(u,**kwargs)

        #u = self.average_pool(u)
        return u

    def get_total_propagation_distance(self):
        z = 0


    def get_propagation_fields(self,Input):
        u = tf.cast(Input, dtype = tf.complex64)
        u_array = []
        u_array.append(u)
        for layer in self.elements:
            #print(layer.name)
            fields = layer.call(u)
            u = layer.call(u)
            u_array = u_array + [fields]
        return u_array
    
    def get_input_size(self):
        return self.get_input_shape()
    

    def get_output_image(self, input):
        return self.call(input)


    def get_propagation_fields_anim(self,Input):
        u = tf.cast(Input, dtype = tf.complex64)
        u_array = []
        u_array.append(u)
        for layer in self.elements:
            print(layer.name)
            fields = layer.call_for_fields(u)
            u = layer.call(u)
            #u = u + u
            #u_array.append(u.numpy())
            u_array = u_array + fields
        return u_array


    def save_field_animation(self, fields, PATH):
        anim = get_field_animation(fields,10, True)
        save_gif_animation(anim, PATH+'/PropagationAnimation.gif')

    def save_fields(self,fields,PATH, anim_fields = None, save_animations = False):
        with open(PATH+'/PropagationFields.pkl', 'wb') as f:
            pkl.dump(fields, f)
        if anim_fields:
            self.save_field_animation(anim_fields, PATH)
        #np.save(PATH+'/PropagationFields.npy',np.array(fields))
        for i in range(0, len(fields)):
            if len(fields[i].shape) == 4:
                fig, axs = plt.subplots(2, fields[i].shape[3], figsize=(10, 10))
                if fields[i].shape[3] > 1:
                    for j in range(0, fields[i].shape[3]):
                    # ax1.imshow(np.abs(model_output[i,:,:,0]))
                        im1 = axs[0,j].imshow(np.abs(fields[i][0, :, :, j]))
                        axs[0,j].axis('off')
                        axs[0,j].set_title('Output (Abs)')
                        fig.colorbar(im1, ax=axs[0,j])

                        im2 = axs[1,j].imshow(np.angle(fields[i][0, :, :, j]))
                        axs[1,j].axis('off')
                        axs[1,j].set_title('Output (phase)')
                        fig.colorbar(im2, ax=axs[1,j])
                        fig.savefig(PATH + '/Propagation_field_{}.png'.format(i))
                        if save_animations:
                            anim = get_polarization_animation(fields[i][0], (30,30), 10.0, True)
                            save_gif_animation(anim, PATH+'/Propagation_field{}_animation.gif'.format(i))

                else:
                    im1 = axs[0].imshow(np.abs(fields[i][0, :, :, 0]))
                    axs[0].axis('off')
                    axs[0].set_title('Output (Abs)')
                    fig.colorbar(im1, ax=axs[0])

                    im2 = axs[1].imshow(np.angle(fields[i][0, :, :, 0]))
                    axs[1].axis('off')
                    axs[1].set_title('Output (phase)')
                    fig.colorbar(im2, ax=axs[1])
                    fig.savefig(PATH + '/Propagation_field_{}.png'.format(i))
                plt.close(fig)
        return


    #def trim_propagation_batch_size(self, fields, batch_size = 1):
    #    for indx, field in enumerate(fields):
    #        fields[indx] = field[0:batch_size,:,:,:]
    #    return fields



    def save_propagation_fields(self, Input, PATH):
        def trim_input( input, n=1):
            return input[0:n]
        
        input_fields = trim_input(Input)
        anim_fields = self.get_propagation_fields_anim(input_fields)
        save_fields = self.get_propagation_fields(input_fields)
        #fields = trim_propagation_batch_size(fields)
        #self.save_field_animation(anim_fields,)
        self.save_fields(save_fields,PATH, anim_fields)


    def get_image_variables(self):
        res = {}
        for i in range(0,len(self.elements)):
            el = self.elements[i].get_image_variables()
            name = self.elements[i].name
            if el != None:
                #res.append(el)
                for k in el.keys():
                    res[name +'/'+ k] = el[k]

        return res
    
    def get_plate_positions(self):
        z = 0.0
        dist_arr = []
        for layer in self.elements:
            if hasattr(layer, 'z'):
                z = z + layer.z
            if isinstance(layer, OL.PhasePlate) or isinstance(layer, OL.PolarizationRotationPlate) or isinstance(layer, OL.JonesPhasePlate ):
                dist_arr.append(z)
        return dist_arr


    def save_phase_plates_to_numpy(self, file):
        '''
        Saves the phase plates (only phases) to a numpy file
        :param file: The string of the numpy file
        :return:
        '''
        phase_plates = []
        #print(self.weights.shape)
        for layer in self.weights:
            phase_plates.append(layer.numpy())
        #phase_plates = np.array(phase_plates)
        #print(phase_plates.shape)
        np.save(file, phase_plates)
        return

    def save_phase_plates_to_images(self, file):
        '''
        Saves the phase plates (only phases) to an image file
        '''
        #phase_plates = []
        phase_plates = self.get_image_variables()
        #i = 0
        for i in range(0,phase_plates.shape[0]):
            plt.figure()
            plt.imshow(phase_plates[i,:,:])
            plt.savefig(file+'/PhasePlate_{}.png'.format(i))
            plt.close()
            #i = i +1

