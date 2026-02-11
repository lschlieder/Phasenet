import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.PropagationLayer import PropagationLayer
from PhaseplateNetwork.utils.util import repeat_2d_tensor


class PhasePropagationLayer( tf.keras.layers.Layer):
    '''
    PhasePropagationLayer class
    Calculates the propagation of a pressure field to a certain distance followed by a multiplication with a trainable complex number ( a phase plate)
    I~~>~~P~~>~~~|*H
    '''
    def __init__(self, z, u, trainable_pixel = 100, scale = 2, L = 80, padding = 50, PhaseTrainable= True, AmplitudeTrainable= False,f0 = 2.00e6, cM = 1484, use_FT = True, use_tanh_activation = True, jitter = False, **kwargs):
        '''
        PhasePropagationLayer class for propagation first propagates a pressure field and than multiplies it with a trainable complex number (phase plate)
        :param z: Distance in [mm]
        :param u: Initialization value for the Phase of the phaseplate with dimension (Nplate, Nplate)
        :param trainable_pixel: the number of trainable pixel on the trainable plate
        :param scale: the scale with which the trainable pixel are multiplied with
        :param L: The size of the trainable plate in [mm]
        :param padding: the padding for the propagation in pixels
        :param PhaseTrainable: Set the Phase part of the phaseplate to be trainable
        :param AmplitudeTrainable: Set the amplitude part of the phaseplate to be trainable
        :param f0: Frequency in 1/s
        :param cM: Wavespeed in medium in [mm]/s
        :param use_FT: Use Fourier transform instead or Matrix multiplication for PropagationLayer
        :param use_tanh_activation: Take the tanh of the phase and multiply by pi so that the result is between -pi:pi
        :param jitter: Use a random 1 pixel jitter every propagation so that the network is more robust to experimental errors
        :param kwargs: Pass through arguments
        '''
        super(PhasePropagationLayer, self).__init__(**kwargs)
        assert u.shape[0] == u.shape[1], 'Input shapes must be quadratic'
        assert u.shape[0] == trainable_pixel, 'Input shape is not equal to the number of trainable pixel'
        assert isinstance(scale,int), 'scale must be integer valued'

        self.phase = tf.Variable(u, trainable=PhaseTrainable, dtype=tf.float32)
        amp = np.ones((u.shape[0],u.shape[1]))
        self.amplitude = tf.Variable(amp, trainable=AmplitudeTrainable, dtype=tf.float32)

        self.z = tf.constant(z, dtype = tf.float32)

        self.N_plate = trainable_pixel*scale
        self.N = self.N_plate + padding
        self.N_plate = self.N_plate
        self.L = L
        self.L_prop_area =  self.N * self.L/self.N_plate
        self.scale = scale
        self.jitter = tf.constant(jitter,dtype= tf.bool)

        #print(self.N)
        self.propagation_layer = PropagationLayer(z,self.N,self.L_prop_area,padding,f0,cM, True,use_FT)
        self.use_tanh = use_tanh_activation

    def build(self,input_shape):
        return



    @tf.function
    def call(self, input):
        '''
        Performs the propagation with the propagation layer and the multiplication
        :param input: Input images of dimension (batch, N ,N , 1)
        :return: Propagated and multiplied image
        '''
        u = input

        u = self.propagation_layer(u)
        u_before = u

        #phase = tf.reshape(self.phase, (self.phase.shape[0], self.phase.shape[1], 1))
        #scaled_phase = tf.squeeze(tf.image.resize(phase, (self.N_plate, self.N_plate)))
        scaled_phase = repeat_2d_tensor(self.phase,self.scale)

        ##amplitude = tf.reshape(self.amplitude,( self.amplitude.shape[0], self.amplitude.shape[1], 1))
        #scaled_amplitude = tf.squeeze(tf.image.resize( amplitude, ( self.N_plate, self.N_plate)))
        scaled_amplitude = repeat_2d_tensor(self.amplitude,self.scale)

        diff = self.N - self.N_plate
        complete_phase = tf.squeeze(tf.pad(scaled_phase, [[diff//2, diff//2 ],[diff//2, diff//2]], constant_values= 0.0))

        complete_amplitude = tf.squeeze(tf.pad( scaled_amplitude, [[diff//2, diff//2 ],[diff//2, diff//2]], constant_values = 0.0))
        if self.use_tanh:
            modulated_phase = tf.math.tanh(complete_phase) * tf.constant(np.pi)
        else:
            modulated_phase = complete_phase
        phase_plate = tf.complex(complete_amplitude, 0.0) * tf.math.exp(tf.complex(0.0, modulated_phase))

        #print(self.jitter)
        if self.jitter:
            r_x = tf.random.uniform([],-1,1,dtype = tf.int32)
            r_y = tf.random.uniform([],-1,1,dtype = tf.int32)
            phase_plate = tf.roll(tf.roll(phase_plate, shift = r_x, axis = 0), shift = r_y, axis = 1)

        phase_plate = tf.reshape(phase_plate, (self.N, self.N, 1))

        u = u * phase_plate

        return u, u_before

    def inverse_call(self, input):

        u = input
        u = self.remove_phase_shift(u)
        u = self.propagation_layer.inverse_call(u)
        return u

    def remove_phase_shift(self, output):
        '''
        Removes the phaseplate multiplication
        :param output: the output of the 'call' function
        :return: The incoming wavefield to the phaseplate
        '''
        u = output
        if self.use_tanh:
            modulated_phase = tf.math.tanh(self.complete_phase)* tf.constant(np.pi)
        else:
            modulated_phase = self.complete_phase
        phase_plate = tf.complex(self.complete_amplitude,0.0) * tf.math.exp(tf.complex(0.0,modulated_phase))
        phase_plate = tf.reshape( phase_plate,(self.N,self.N,1))

        u = u / phase_plate
        return u