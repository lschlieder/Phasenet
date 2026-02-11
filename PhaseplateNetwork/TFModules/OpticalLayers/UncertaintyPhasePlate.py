import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
from PhaseplateNetwork.utils.util import repeat_image_tensor
import tensorflow_probability as tfp


EPSILON = 1e-16
class UncertaintyPhasePlate(OpticalLayer):
    def __init__(self, shape, scale = 2, amplitude_trainable = False, phase_trainable = True, sigma_amp = None, sigma_phase = 0.01, initialization = 'zeros', **kwargs):
        '''
        A layer to multiply the incoming complex field with a phase plate that has (possible trainable) amplitude and phase with uncertainty
        shape: the input shape of the image
        scale: upscale the plate
        amplitude_trainable: weather the amplitude of the phaseplate is trainable
        sigma: the sigma of the added gaussian i.i.d. noise
        phase_trainable: makes the phase of the phaseplate trainable (enabled by default)
        initialization: The initialization parameter given the the phase part of the phaseplate. defaults to zero, since random values gives essentially a diffuser and give no good gradient
        '''
        super(UncertaintyPhasePlate, self).__init__(**kwargs)
        #if np.array(plate_size).size == 1 :
        #    self.N = np.array([plate_size,plate_size])
        #else:
         #   self.N = np.array(plate_size)
        self.scale = scale
        self.plate_shape = shape
        self.amplitude_trainable= amplitude_trainable
        self.phase_trainable = phase_trainable
        self.sigma_phase = sigma_phase
        self.sigma_amp = sigma_amp

        self.phaseplate_shape = [1, shape[0], shape[1],1]
        if initialization =='zeros':
            init = np.zeros(self.phaseplate_shape)
        elif initialization == 'uniform':
            init = np.random.uniform(0.0, 2*np.pi/20, size = self.phaseplate_shape)
        self.amplitudes = tf.Variable(np.ones(self.phaseplate_shape), trainable = amplitude_trainable, dtype = tf.float32)
        self.phases = tf.Variable(init, trainable= phase_trainable,dtype = tf.float32)
        if self.sigma_phase != None:
            self.n_dist_phase = tfp.distributions.Normal(loc = 0.0, scale = self.sigma_phase)
        if self.sigma_amp != None:
            self.n_dist_amp = tfp.distributions.Normal(loc = 0.0, scale = self.sigma_amp)

        



    def call(self, Input, training = None, **kwargs):
        assert( Input.shape[1] >= self.phases.shape[1]*self.scale)
        assert( Input.shape[2] >= self.phases.shape[2]*self.scale)
        #assert( Input.shape[3] == self.phases.shape[3])
        diffx = Input.shape[1] - self.phases.shape[1]*self.scale
        diffy = Input.shape[2] - self.phases.shape[2]*self.scale
        #diff_upx = tf.math.round(diffx/2)
        diff_upx= tf.cast(tf.math.round(diffx/2),dtype = tf.int32)
        diff_upy = tf.cast(tf.math.round(diffy/2), dtype = tf.int32)
        diff_downx = tf.cast(tf.math.round(diffx/2 ), dtype = tf.int32)
        diff_downy = tf.cast(tf.math.round(diffy/2),dtype = tf.int32)




        amp = tf.pad(repeat_image_tensor(self.amplitudes,self.scale), [ [0,0],[diff_upx, diff_downx], [diff_upy, diff_downy], [0,0]], "CONSTANT",1.0)

        #amp = tf.image.resize(self.amplitudes, (Input.shape[1], Input.shape[2]), method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        phase = tf.pad(repeat_image_tensor(self.phases, self.scale), [ [0,0],[diff_upx, diff_downx], [diff_upy, diff_downy], [0,0]], "CONSTANT",0.0)
        if training:
            if self.sigma_phase != None:
                noise_phase = tf.cast(self.n_dist_phase.sample(phase.shape), tf.float32)
                phase = phase + noise_phase
            if self.sigma_amp != None:
                noise_amp = tf.cast(self.n_dist_amp.sample(amp.shape), tf.float32)
                amp = amp + noise_amp

        phase_plate = tf.complex(amp + EPSILON, 0.0) * tf.cast(
            tf.exp(1j * tf.cast(phase, dtype=tf.complex64)), dtype=tf.complex64)


        new_field = Input * phase_plate
        return new_field

    def compute_output_shape(self, input_shape):
        #assert( input_shape[1] == self.phases.shape[1])
        #assert( input_shape[2] == self.phases.shape[2])
        #assert( input_shape[3] == self.phases.shape[3])
        output_shape = []
        for i in range(0,len(input_shape)):
            output_shape.append(input_shape[i])
        return output_shape

    def get_image_variables(self):
        if len(self.trainable_variables) != 0:
            #print(self.trainable_variables[0])
            #print(len(self.trainable_variables))
            res = tf.concat(self.trainable_variables, axis = 0)
            res = res[:,:,:,0]
        else:
            res = None
        #print(res.shape)

        return res

    def get_config(self):
        #shape, scale = 2, amplitude_trainable = False, phase_trainable = True
        temp = {
            'shape': self.plate_shape,
            'scale': self.scale,
            'amplitude_trainable': self.amplitude_trainable,
            'phase_trainable': self.phase_trainable,
            'sigma': self.sigma
        }
        return temp










