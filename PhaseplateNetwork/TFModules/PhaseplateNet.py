import tensorflow as tf
#from hologram import Hologram
import numpy as np
#import PhasePropagationLayer, PropagationLayer
from PhaseplateNetwork.TFModules.PhasePropagationLayer import PhasePropagationLayer
from PhaseplateNetwork.TFModules.PropagationLayer import PropagationLayer
import matplotlib.pyplot as plt
# Phaseplate net
#from ang_spec import propagate_field_AS_spectral
#from PhaseplateNetwork.util import repeat, repeat_2d_tensor
from PhaseplateNetwork.utils.util import repeat, repeat_2d_tensor



def jitter(inp):
    r_x = tf.random.uniform([],-1,1,dtype = tf.int32)
    r_y = tf.random.uniform([],-1,1,dtype = tf.int32)
    inp = tf.roll(tf.roll(inp, shift = r_x, axis = 1), shift = r_y, axis = 2)
    return inp

############################

class PhaseplateNet(tf.keras.Model):
    def __init__(self, z, trainable_pixel = 100, scale = 2, L = 80, padding = 50, f0 = 2.00e6, cM = 1484, DH = 40, first_layer_distance = 30, last_layer_distance = 30, use_FT = True, input = 'amplitude', add_evanescence_layers = False, use_fourier_lens = False, use_image_as_direct_input = False, use_tanh_activation = False, jitter = False, activation = None ):
        '''
        PhasePlate Network consisting of multiple Phaseplates and propagation layers
        :param z: List of distances for the network ( i.e. [30,30,30])
        :param trainable_pixel: Number of trainable pixels that are upscaled by scale
        :param scale: integer number to upscale the Trainable pixels to get better propagation images
        :param padding: padding in pixels which is added at the side of the image to avoid reflection issues.
        :param f0: Frequency in [1/s]
        :param cM: Speed in Medium in [mm/s]
        :param DH: Transducer Diameter
        :param first_layer_distance: propagation distance from the Transducer to the input Plate
        :param last_layer_distance: Propagation distance from the last plate to the output plane
        :param use_FT: Use the angular spectrum method to propagate the wavefields. If False use Matrix multiplication
        :param amplitude_input: Use amplitude instead of phase as input plate
        :param add_evanescence_layers: Add another layer shortly behind every layer ( deprecated since no evansecent propagation takes place anymore)
        :param use_fourier_lens: Add one plate with a fourier lens before the first and after the second to calculate in k-space
        :param use_image_as_direct_input: No Transducer propagation. Use the amplitude of the input as the first layer
        :param use_tanh_activation: Train the phaseplates with pi*tanh instead of just direct training for a constriction of the values to [-pi:pi]
        :param jitter: Use random 1pixel jitter to make the network robust against experimental errors
        :param activation: Use nonlinearity after each layer ( 'relu', 'abs', 'none'/None)
        '''
        super(PhaseplateNet, self).__init__()
        self.scale = scale
        assert isinstance(scale, int), "scale must be integer like"
        self.N_plate = int(trainable_pixel * scale)
        self.N = self.N_plate + padding
        self.Trainable_Pixel = trainable_pixel
        self.f0 = f0
        self.DH = DH
        self.cM = cM
        if (L != DH*2):
            print('Warning L != DH*2. Plate Size L = {}, Transducer size DH = {}'.format(L,DH))
        self.L = L
        self.L_prop_area = self.N * self.L/self.N_plate
        #self.amplitude_input = amplitude_input
        self.input_datatype = input
        self.activation = activation
        self.use_image_as_direct_input = use_image_as_direct_input
        self.jitter = tf.constant(jitter, dtype = tf.bool)
        if self.L < DH:
            print('Warning: L < DH. are you sure you want to do this?' )
        wavelength = cM / f0 * 1000

        self.dp = self.L / (self.Trainable_Pixel - 1)
        self.dp_im = self.L/(self.Trainable_Pixel* scale)
        [X, Y] = np.meshgrid((np.arange(-self.N/2, self.N/2, 1))*self.dp_im, (np.arange(-self.N/2, self.N/2, 1))*self.dp_im)
        [R, Th] = cart2pol(X,Y)
        self.df = 1/self.L
        [self.fx, self.fy] = np.meshgrid((np.arange(-self.N/2, self.N/2, 1))*self.df, (np.arange(-self.N/2, self.N/2, 1))*self.df)
        self.fmax = 1/wavelength
        print('transducer_radius: {}, dp: {}'.format(self.DH*(self.L/self.N), self.dp_im))
        #self.transducer_amplitude = 1.0 * (R < self.DH* (self.L/self.N))
        self.transducer_amplitude = 1.0*(R < self.DH)

        self.kx = (2*np.pi*self.fx).astype('float32')
        self.ky = (2*np.pi*self.fy).astype('float32')
        self.ka = np.float(2 * np.pi / wavelength)
        self.f_print_max = 1/(2*self.dp)

        self.phi_max = np.arcsin(wavelength * self.f_print_max)

        self.use_fourier_lens = use_fourier_lens
        self.use_tanh_activation = use_tanh_activation

        #self.N_plate = int(self.N * self.Pl/self.L)


        self.phase_plates = []
        self.phase_weights = []
        self.fully_connected_distance = self.L/(np.tan(self.phi_max))   # use the complete size PL because even the pixels on the bottom need to propagated to those on the top

        self.connection_size = np.tan(self.phi_max) * np.array(z)
        self.connection_fraction = self.connection_size / self.L
        print('fully connected distance: {}'.format(self.fully_connected_distance))

        print(self.fully_connected_distance)
        self.first_layer_distance = first_layer_distance
        self.last_layer_distance = last_layer_distance
        self.u_array = []

        #self.offset = 0.1
        self.offset = tf.Variable(0.4, trainable = False)
        ###################### fourier lens
        if self.use_fourier_lens:
            focus = self.fully_connected_distance
            [X, Y] = np.meshgrid((np.arange(-self.N // 2 +1 , self.N // 2+1 , 1)) * self.dp_im,
                                 (np.arange(-self.N // 2 +1 , self.N // 2+1 , 1)) * self.dp_im)
            X_squared = X ** 2
            Y_squared = Y ** 2
            phase = -(np.pi / (wavelength * focus) * (X_squared + Y_squared)).astype('float32') % (2 * np.pi) - np.pi
            phase = 0.5 * ( np.log( (1+ phase/np.pi)/(1-phase/np.pi)))
            self.fourier_layer = PhasePropagationLayer(self.fully_connected_distance, phase,trainable_pixel,scale,L,padding, False,False, f0 = f0, cM = cM, use_FT = use_FT,use_tanh_activation= use_tanh_activation, jitter = jitter)
            self.inverse_fourier_layer = PhasePropagationLayer(self.fully_connected_distance, phase, False,False, self.L, self.scale, self.N_plate, f0 = f0, cM = cM, use_FT = use_FT,use_tanh_activation= use_tanh_activation, jitter = jitter)


        #################################################################################################################
        ############    Layer Setup
        #################################################################################################################
        print('setting up propagation layers')

        self.first_propagation = PropagationLayer(first_layer_distance,self.N,self.L_prop_area,padding,f0,cM,channels_last = True, use_FT=use_FT)
        self.last_propagation = PropagationLayer(last_layer_distance,self.N,self.L_prop_area,padding,f0,cM,channels_last = True, use_FT=use_FT)


        print('setting up phase plates')
        self.phase_propagation_layers = []
        self.z = z
        for z_i in z:
            if add_evanescence_layers:
                self.phase_propagation_layers.append(
                    PhasePropagationLayer(0.1, np.zeros((trainable_pixel, trainable_pixel)),trainable_pixel,scale, L, padding, True, False, f0 = f0, cM = cM, use_FT = use_FT,use_tanh_activation = use_tanh_activation, jitter = self.jitter))
                self.phase_propagation_layers.append(
                    PhasePropagationLayer(z_i, np.zeros((trainable_pixel,trainable_pixel)),trainable_pixel,scale, L, padding, False,False, f0 = f0, cM = cM, use_FT = use_FT,use_tanh_activation = use_tanh_activation, jitter = self.jitter )
                )

            else:
                self.phase_propagation_layers.append(
                    PhasePropagationLayer(z_i, np.zeros((trainable_pixel, trainable_pixel)),trainable_pixel,scale, L, padding,True, False, f0 = f0, cM = cM, use_FT = use_FT,use_tanh_activation = use_tanh_activation, jitter = self.jitter))



    def build(self,input_shape):
        #print(input_shape)
        amplitude = tf.cast( tf.reshape( self.transducer_amplitude, (self.N,self.N,1)),tf.float32)
        amplitude = tf.complex(amplitude, tf.zeros(amplitude.shape, dtype=tf.float32))
        self.amplitude = amplitude

    #@tf.function
    def get_input_phase_plate(self,input):
        '''
        returns the input phase plate depending on self.amplitude_input
        :param input: input image of dimension (batch, N,N, 1) and dtype float32
        :return: The complex input
        '''
        if self.input_datatype == 'amplitude':
            u = tf.complex(input,0.0)*tf.exp(tf.complex(0.0, tf.zeros(input.shape)))
        elif self.input_datatype == 'complex':
            u = input
        elif self.input_datatype == 'phase':
            u = tf.exp(tf.complex(0.0, np.pi * input))
        else:
            raise Exception('input type not know: {}'.format(self.input))
        return u

    #@tf.function
    def get_input_image(self,input, u_array = None):
        '''
        Get the input image depending on the first propagation and the amplitude input option
        :param input: Input image of dimension (batch, N,N,1) and dtype float32
        :param u_array: (optional) list to put the propagations into to return them as result for a network call for debugging
        :return: Returns the "input" image which is the first image which is the input of the actual trainable phase plates
        '''
        if self.use_image_as_direct_input:
            u = self.get_input_phase_plate(input)
        else:
            inp_amp = self.amplitude * tf.exp(tf.complex(0.0, tf.zeros(input.shape)))

            prop_inp = self.first_propagation(inp_amp)
            if u_array != None:
                u_array.append(inp_amp)
                u_array.append(prop_inp)
            phase_plate = self.get_input_phase_plate(input)
            #print(self.jitter)
            if self.jitter:
                r_x = tf.random.uniform([], -1, 1, dtype=tf.int32)
                r_y = tf.random.uniform([], -1, 1, dtype=tf.int32)
                phase_plate = tf.roll(tf.roll(phase_plate, shift=r_x, axis=1), shift=r_y, axis=2)
            u = prop_inp * phase_plate
        return u


    @tf.function
    def call(self, inputs, **kwargs):
        '''
        Calls the network on a batch of input images
        :param inputs: input images of dimension (batch, N,N,1) and dtype float32
        :param kwargs: Keras options
        :return: Returns the propagated images and the propagation steps
        '''
        u = inputs
        u_array = []
        u = self.get_input_image(u,u_array)
        if self.jitter:
            r_x = tf.random.uniform([],-1,1,dtype = tf.int32)
            r_y = tf.random.uniform([],-1,1,dtype = tf.int32)
            u = tf.roll(tf.roll(u, shift = r_x, axis = 1), shift = r_y, axis = 2)

        if self.use_fourier_lens:
            u_array.append(u)
            u = self.fourier_layer(u)
            u_array.append(u)
            for i in range(0,2):
                u = self.phase_propagation_layers[i](u)
                u_array.append(u)
            u = self.inverse_fourier_layer(u)
            u_array.append(u)
            for i in range(2,len(self.phase_propagation_layers)):
                u = self.phase_propagation_layers[i](u)
                u_array.append(u)

        else:
            u_array.append(u)
            for i in range(0,len(self.phase_propagation_layers)):
                u, before_u = self.phase_propagation_layers[i](u)
                if self.activation == 'relu':
                    print('use relu nonlinearity')
                    mask = tf.cast(tf.math.abs(u) >= self.offset, dtype = tf.complex64)
                    not_mask = tf.cast(tf.abs(mask-1.0),dtype = tf.complex64)
                    u = mask * u + 0.1 * u * not_mask

                elif self.activation == 'abs':
                    u = tf.complex(tf.math.abs(u), 0.0) * tf.math.exp(tf.complex(0.0,0.0))
                    print('use abs nonlinearity')
                elif self.activation == 'tanh':
                    u = tf.math.sigmoid(u)
                elif self.activation =='log':
                    u = tf.cast(tf.math.log(tf.math.abs(u)+1.0)+0.000000000001,dtype=tf.complex64) * tf.math.exp(1j*tf.cast(tf.math.angle(u),dtype=tf.complex64))



                elif self.activation == 'none' or self.activation is None:
                    u = u


                u_array.append(before_u)
                u_array.append(u)


        u = self.last_propagation(u)

        return u, u_array

    def inverse_call(self,input):
        u = input
        u_array = []

        u = self.last_propagation.inverse_call(u)

        u_array.append(u)


        for i in range(0, len(self.phase_propagation_layers)):
            u = self.phase_propagation_layers[len(self.phase_propagation_layers) - i -1 ].inverse_call(u)
            u_array.append(u)
        return u, u_array


    def save_phase_plates_to_numpy(self, file):
        '''
        Saves the phase plates (only phases) to a numpy file
        :param file: The string of the numpy file
        :return:
        '''
        phase_plates = []
        for layer in self.phase_propagation_layers:
            print(np.max(layer.weights[0].numpy()))
            print(np.min(layer.weights[0].numpy()))
            if self.use_tanh_activation:
                phase_plate = (np.tanh(layer.weights[0].numpy())*np.pi)
            else:
                phase_plate = (layer.weights[0].numpy())%(2*np.pi)
            # plt.imshow(phase_plate)
            # plt.show()
            phase_plates.append(phase_plate)

        phase_plates = np.array(phase_plates)
        np.save(file, phase_plates)
        return

    def get_network_hyperparameters(self):
        '''
        Get the network hyperparameters
        :return: This returns all parameters that are needed to reproduce the network in the real world. Training options are not returned
        '''
        str =       'Network hyperparameters and physical dimensions:\n'
        str = str + '\n'
        #str = str + 'Propagation area size: {} mm\n'.format(self.L)
        str = str + 'Plate size: {} mm\n'.format(self.L)
        #str = str + 'Complete Pixel count (pixel in propagation area): {}\n'.format(self.N)
        str = str + 'Image pixel count (pixel on plate): {}\n'.format(self.Trainable_Pixel)
        #str = str + 'Trainable pixel on plate (are upscaled to match pixel on plate): {}\n'.format(self.Trainable_Pixel)
        str = str + 'Trainable pixel size: {} mm\n'.format(self.dp)
        str = str + 'Image pixel size: {} mm\n'.format(self.dp_im)
        str = str + 'Transducer radius: {} mm\n'.format(self.DH)
        str = str + 'Transducer frequency: {} Hz\n'.format(self.f0)
        str = str + 'Wavespeed in medium: {} mm/s\n'.format(self.cM)
        str = str + 'Distance between transducer and first plate: {}\n'.format(self.first_layer_distance)
        str = str + 'Distance between last plate and detector: {}\n'.format(self.last_layer_distance)
        str = str + 'Plate Distance:\n'
        i = 1
        for dist in self.z:
            str = str + 'layer{}: {} with connection fraction: {}\n'.format(i,dist, self.connection_fraction[i-1])
            i = i+1
        i = 1

        str = str +'Fully connected distance: {}\n \n'.format(self.fully_connected_distance)
        return str



    def perform_projection_steps(self, inp, wanted_output):
        layers = len(self.phase_propagation_layers)
        for i in range(0, len(self.phase_propagation_layers)):
            result, u_array = self.call(inp)
            wanted_result = tf.complex( wanted_output,0.0)* tf.math.exp(tf.complex(0.0, tf.math.angle(result)))
            test = tf.math.exp(tf.complex(0.0,tf.math.angle(result)))
            test = tf.complex(wanted_output,0.0) * test

            inverse_result, inverse_u_array = self.inverse_call(wanted_result)


            phases = tf.math.angle(inverse_u_array[i]) - tf.math.angle(self.phase_propagation_layers[layers-i-1].remove_phase_shift(u_array[layers - i]))#* tf.abs(inp[:,:,:,:])

            avg_phase = tf.reshape(tf.reduce_mean(phases,axis = 0),(phases.shape[1], phases.shape[2]))

            self.phase_propagation_layers[layers-i -1].phase.assign(avg_phase)



    #@tf.function
    def perform_single_wavefront_matching_step(self, inp, wanted_output):
        for i in range(0, len(self.phase_propagation_layers)):
            result_inv, u_array_inv = self.inverse_call(tf.complex(wanted_output,0.0))
            result, u_array = self.call(inp)
            phase_removed = self.phase_propagation_layers[i].remove_phase_shift(u_array[i+1])
            mul = phase_removed * tf.math.conj(u_array_inv[i])
            new_phase = tf.reshape( tf.reduce_mean(tf.math.imag(mul), axis = 0 ), (mul.shape[1],mul.shape[2]))
            new_phase_real = tf.reshape( tf.reduce_mean(tf.math.real(mul), axis = 0), (mul.shape[1], mul.shape[2]))

            new_phase = new_phase / new_phase_real


            self.phase_propagation_layers[i].phase.assign(new_phase )


        layer_num = len(self.phase_propagation_layers)-1


        for i in range(0, len(self.phase_propagation_layers)):
            result_inv, u_array_inv = self.inverse_call(tf.complex(wanted_output,0.0))
            result, u_array = self.call(inp)
            phase_removed = self.phase_propagation_layers[layer_num-i].remove_phase_shift(u_array[layer_num-i+1])
            mul = phase_removed * tf.math.conj(u_array_inv[layer_num- i])
            new_phase = tf.reshape( tf.reduce_mean(tf.math.imag(mul), axis = 0 ), (mul.shape[1],mul.shape[2]))
            new_phase_real = tf.reshape( tf.reduce_mean(tf.math.real(mul), axis = 0), (mul.shape[1], mul.shape[2]))

            new_phase = tf.math.atan(new_phase / new_phase_real)
            new_phase = tf.math.atanh(new_phase/(np.pi))

            self.phase_propagation_layers[layer_num-i].phase.assign(new_phase)








'''
@tf.function
def get_propagation_operator(ka, kx, ky, z):
    kz_real = tf.multiply(ka,ka) - tf.multiply(kx,kx) - tf.multiply(ky,ky)
    kz_real_2 = tf.nn.relu(kz_real)
    #if
    #z = tf.reshape(z, (z.shape[0],1,1))
    #print(z.dtype)
    #print(kz_real_2.dtype)
    propagator_complex = tf.math.exp( tf.complex( tf.constant(0.0,shape= (kz_real_2.shape[0],kz_real_2.shape[1])), z* tf.sqrt(kz_real_2) ) )
    #z = tf.reshape(z, [z.shape[0]] )
    return propagator_complex
'''
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)



@tf.function
def get_detector_sum(image,size = 10):
    N = image.shape[1]
    c = np.zeros((10, 2), 'int32')
    c[0, 0], c[0, 1] = int(120 / 512 * N), int(90 / 512 * N)
    c[1, 0], c[1, 1] = int(270 / 512 * N), int(90 / 512 * N)
    c[2, 0], c[2, 1] = int(420 / 512 * N), int(90 / 512 * N)
    c[3, 0], c[3, 1] = int(120 / 512 * N), int(220 / 512 * N)
    c[4, 0], c[4, 1] = int(270 / 512 * N), int(220 / 512 * N)
    c[5, 0], c[5, 1] = int(420 / 512 * N), int(220 / 512 * N)
    c[6, 0], c[6, 1] = int(120 / 512 * N), int(320 / 512 * N)
    c[7, 0], c[7, 1] = int(279 / 512 * N), int(320 / 512 * N)
    c[8, 0], c[8, 1] = int(420 / 512 * N), int(320 / 512 * N)
    c[9, 0], c[9, 1] = int(220 / 512 * N), int(470 / 512 * N)
    c[:, [0, 1]] = c[:, [1, 0]]

    zero_sum = tf.reduce_sum(
        tf.slice(image, (0, c[0, 0] - size, c[0, 1] - size, 0), (image.shape[0], 2 * size, 2 * size, 1)), axis=[1, 2])
    one_sum = tf.reduce_sum(
        tf.slice(image, (0, c[1, 0] - size, c[1, 1] - size, 0), (image.shape[0], 2 * size, 2 * size, 1)), axis=[1, 2])
    two_sum = tf.reduce_sum(
        tf.slice(image, (0, c[2, 0] - size, c[2, 1] - size, 0), (image.shape[0], 2 * size, 2 * size, 1)), axis=[1, 2])
    three_sum = tf.reduce_sum(
        tf.slice(image, (0, c[3, 0] - size, c[3, 1] - size, 0), (image.shape[0], 2 * size, 2 * size, 1)), axis=[1, 2])
    four_sum = tf.reduce_sum(
        tf.slice(image, (0, c[4, 0] - size, c[4, 1] - size, 0), (image.shape[0], 2 * size, 2 * size, 1)), axis=[1, 2])
    five_sum = tf.reduce_sum(
        tf.slice(image, (0, c[5, 0] - size, c[5, 1] - size, 0), (image.shape[0], 2 * size, 2 * size, 1)), axis=[1, 2])
    six_sum = tf.reduce_sum(
        tf.slice(image, (0, c[6, 0] - size, c[6, 1] - size, 0), (image.shape[0], 2 * size, 2 * size, 1)), axis=[1, 2])
    seven_sum = tf.reduce_sum(
        tf.slice(image, (0, c[7, 0] - size, c[7, 1] - size, 0), (image.shape[0], 2 * size, 2 * size, 1)), axis=[1, 2])
    eight_sum = tf.reduce_sum(
        tf.slice(image, (0, c[8, 0] - size, c[8, 1] - size, 0), (image.shape[0], 2 * size, 2 * size, 1)), axis=[1, 2])
    nine_sum = tf.reduce_sum(
        tf.slice(image, (0, c[9, 0] - size, c[9, 1] - size, 0), (image.shape[0], 2 * size, 2 * size, 1)), axis=[1, 2])


@tf.function
def get_classification_vector(image,size = 10):
    N = image.shape[1]
    c = np.zeros((10,2),'int32')
    c[0,0], c[0,1] = int(120 / 512 * N), int(90 / 512 * N)
    c[1,0], c[1,1] = int(270 / 512 * N), int(90 / 512 * N)
    c[2,0], c[2,1] = int(420 / 512 * N), int(90 / 512 * N)
    c[3,0], c[3,1] = int(120 / 512 * N), int(220 / 512 * N)
    c[4,0], c[4,1] = int(270 / 512 * N), int(220 / 512 * N)
    c[5,0], c[5,1] = int(420 / 512 * N), int(220 / 512 * N)
    c[6,0], c[6,1] = int(120 / 512 * N), int(320 / 512 * N)
    c[7,0], c[7,1] = int(279 / 512 * N), int(320 / 512 * N)
    c[8,0], c[8,1] = int(420 / 512 * N), int(320 / 512 * N)
    c[9,0], c[9,1] = int(220 / 512 * N), int(470 / 512 * N)
    c[:,[0,1]] = c[:,[1,0]]

    zero_sum = tf.reduce_sum(
        tf.slice(image, (0, c[0,0] - size, c[0,1] - size, 0), (image.shape[0], 2* size, 2* size, 1)),axis = [1,2] )
    one_sum = tf.reduce_sum(
        tf.slice(image, (0, c[1,0] - size, c[1,1] - size, 0), (image.shape[0], 2* size, 2* size, 1)), axis=[1, 2])
    two_sum = tf.reduce_sum(
        tf.slice(image, (0, c[2,0] - size, c[2,1] - size, 0), (image.shape[0], 2* size, 2* size, 1)), axis=[1, 2])
    three_sum = tf.reduce_sum(
        tf.slice(image, (0, c[3,0] - size, c[3,1] - size, 0), (image.shape[0], 2* size, 2* size, 1)), axis=[1, 2])
    four_sum = tf.reduce_sum(
        tf.slice(image, (0, c[4,0] - size, c[4,1] - size, 0), (image.shape[0], 2* size, 2* size, 1)), axis=[1, 2])
    five_sum = tf.reduce_sum(
        tf.slice(image, (0, c[5,0] - size, c[5,1] - size, 0), (image.shape[0], 2* size, 2* size, 1)), axis=[1, 2])
    six_sum = tf.reduce_sum(
        tf.slice(image, (0, c[6,0] - size, c[6,1] - size, 0), (image.shape[0], 2* size, 2* size, 1)), axis=[1, 2])
    seven_sum = tf.reduce_sum(
        tf.slice(image, (0, c[7,0] - size, c[7,1] - size, 0), (image.shape[0], 2* size, 2* size, 1)), axis=[1, 2])
    eight_sum = tf.reduce_sum(
        tf.slice(image, (0, c[8,0] - size, c[8,1] - size, 0), (image.shape[0], 2* size, 2* size, 1)), axis=[1, 2])
    nine_sum = tf.reduce_sum(
        tf.slice(image, (0, c[9,0] - size, c[9,1] - size, 0), (image.shape[0], 2* size, 2* size, 1)), axis=[1, 2])

    res = tf.concat([zero_sum, one_sum, two_sum, three_sum, four_sum, five_sum, six_sum, seven_sum, eight_sum, nine_sum], -1)
    return res

@tf.function
def get_classification_vector_loss(image,wanted_result,size = 10):
    res = get_classification_vector(image,size)
    sum = tf.math.reduce_sum(res)
    loss = tf.nn.softmax_cross_entropy_with_logits(wanted_result, res) + 1/sum
    return loss



@tf.function
def compute_gradients_and_perform_update_step(model, optimizer,inp, wanted_output):
    with tf.GradientTape() as g:
        result, u_array = model(inp)
        result = tf.abs(result)
        #loss = tf.reduce_mean((result-tf.math.reduce_max(result) *wanted_output)**2)
        loss = tf.reduce_mean((result - wanted_output)**2)
    #print(len(model.weights))
    #print(model.weights)


    gradients = g.gradient(loss, model.weights)
    #plt.imshow(gradients[0])
    #plt.show()
    optimizer.apply_gradients(zip(gradients, model.weights))
    #plt.imshow(gradients[0].numpy())
    #plt.show()

    return loss, result, wanted_output, u_array

#@tf.function
def compute_gradients_and_perform_update_step_PONCS(model, optimizer,inp, wanted_output):
    with tf.GradientTape(persistent = True) as g:
        result, u_array = model(inp)
        #loss2 = tf.reduce_sum((tf.abs(tf.abs(result) - np.max(result)*wanted_output)))
        #print(result.shape)
        #print(tf.abs(result).dtype)
        #print(wanted_output.dtype)
        #print(np.max(result).dtype)
        #loss_single = tf.reduce_mean(((tf.abs(result) - tf.math.reduce_max(tf.abs(result))*wanted_output)**2),axis = (1,2,3))
        #loss_single = tf.reduce_mean( ((tf.abs(result) -tf.math.reduce_max(tf.abs(result))* wanted_output)**2), axis = (1,2,3))
        loss_single = tf.reduce_mean( (tf.abs((tf.abs(result) - wanted_output))), axis = (1,2,3))
        loss = tf.reduce_mean((tf.abs(result) - wanted_output)**2)
        loss_s = []
        for m in range(0,len(loss_single)):
           loss_s.append(loss_single[m])

    gradients = []
    for l in loss_s:
        gradients.append(g.gradient(l,model.weights))
    #single_grad = g.gradient(loss,model.weights)
    #single_grad2 = g.gradient(loss2,model.weights)
    #print(len(gradients))
    #print(len(gradients[0]))

    N = len(loss_single)
    #for k in range(0, len(gradients)):

    k = np.random.randint(0,len(gradients))
    layer = np.random.randint(0,len(gradients[k]))
    while None is gradients[k][layer]:
        k = np.random.randint(0, len(gradients))
        layer = np.random.randint(0, len(gradients[k]))
    #print(k)
    #print(layer)
    #print(len(gradients[k]))
    gradients[k][layer] = (loss_s[k])/(tf.reduce_sum(tf.convert_to_tensor(gradients[k][layer])**2) + 0.000000000000001)*gradients[k][layer]/10
    #gradients[k][layer] = 0.001/(tf.reduce_sum(tf.convert_to_tensor(gradients[k][layer])**2) + 0.000000000000001)*gradients[k][layer]
    model.weights[layer].assign( model.weights[layer] - gradients[k][layer])

    new_result, new_u_array = model(inp)
    '''
    plt.imshow(np.abs(result[0,:,:,0]))
    plt.figure()
    plt.imshow(np.abs(new_result[0,:,:,0]))
    plt.figure()
    plt.imshow(wanted_output[0,:,:,0])
    plt.figure()
    plt.imshow(gradients[k][layer][:,:])
    plt.show()
    '''
    '''
    new_weights = [tf.zeros((gradients[0][0].shape))]*len(gradients[0])

    optimizing_plate = np.random.randint(0,len(gradients[0])/2)
    #optimizing_plate =
    #print(gradients)
    for k in range(0, len(gradients)):
        # if((loss/sum) < 100):
        # gradients[k] = ((loss-1.5)/(sum+ 0.0000000001)) * gradients[k]
        new_weights_short = []
        layer = np.random.randint(0,len(gradients[k]))
        for j in range(0,len(gradients[k])):
            #print(optimizing_plate)
            #print(j)
            if j == optimizing_plate:
                if gradients[k][j] != None:
                    #gradients[k] = np.abs(loss) / (sum + 0.0000000001) * gradients[k]/30
                    #plt.figure()
                    #plt.imshow(gradients[k][j])
                    #print(tf.reduce_sum(tf.convert_to_tensor(gradients[k][j])))
                    #print(loss_s[k]/(tf.reduce_sum(tf.convert_to_tensor(gradients[k][j])**2) + 0.00000000001) )
                    #print(loss_s[k])
                    gradients[k][j] = (loss_s[k])/(tf.reduce_sum(tf.convert_to_tensor(gradients[k][j])**2) + 0.00000000001)  * gradients[k][j] /(N)


                    #new_weights_short.append( model.weights[j] - gradients[k][j])
                    #print(gradients[k][j].numpy())

                    #new_weights_2[j] = new_weights[j] + gradients[k][j].numpy()

                    #new_weights.append(gradients[k][j])
                    new_weights[j] = new_weights[j] + gradients[k][j]




                else:
                    new_weights[j] = None

            else:

                new_weights[j]= None
    '''
    #optimizer.apply_gradients(zip(new_weights,model.weights))
    #optimizer.apply_gradients(zip(single_grad,model.weights))
    #plt.imshow(gradients[0].numpy())
    #plt.show()
    del g
    return loss, tf.abs(result), wanted_output, u_array

@tf.function
def compute_gradients_and_perform_update_step_for_logit_classification(model, optimizer,inp, wanted_output):
    with tf.GradientTape() as g:
        result, u_array = model(inp)
        loss = tf.reduce_mean(get_classification_vector_loss(result, wanted_output,4))
        #loss = tf.reduce_mean((result - tf.math.reduce_max(result) * wanted_output)**2)
        #print(np.max((tf.math.reduce_max(result) * wanted_output).numpy()))
        #print(np.min((tf.math.reduce_max(result) * wanted_output).numpy()))
        #print(np.max(result.numpy()))
        #plt.imshow(((result - tf.math.reduce_max(result) * wanted_output)**2)[0,:,:,0].numpy(), vmax=1.0, vmin = -1.0)
        #plt.show()
        #plt.imshow(result[0,:,:,0].numpy())
        #plt.show()
    gradients = g.gradient(loss, model.weights)

    optimizer.apply_gradients(zip(gradients, model.weights))


    return loss, result, wanted_output, u_array

def get_accuracy(result,wanted_output,size = 10):
    val_arr = tf.math.softmax(get_classification_vector(result, size))
    # print(val_arr.shape)
    # print(number.shape)
    ind_res = np.zeros((result.shape[0], 1))
    ind_num = np.zeros((result.shape[0], 1))
    for b in range(0, result.shape[0]):
        # print(np.where(val_arr[b].numpy() == np.max(val_arr[b].numpy())))
        #print(np.where(val_arr[b].numpy() == np.max(val_arr[b].numpy())))
        arr = np.where(val_arr[b].numpy() == np.max(val_arr[b].numpy()))
        if arr[0].shape[0] > 1:
            ind_res[b] = (arr[0][0])
        else:
            ind_res[b] = arr
        #ind_res[b] = np.where(val_arr[b].numpy() == np.max(val_arr[b].numpy()))
        ind_num[b] = np.where(wanted_output[b].numpy() == np.max(wanted_output[b].numpy()))
    # print(ind_res)
    # print(ind_num)
    count_arr = (ind_res == ind_num)
    return np.mean(count_arr), np.sum(count_arr)
