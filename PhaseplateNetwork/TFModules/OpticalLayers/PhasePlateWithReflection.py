import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
from PhaseplateNetwork.utils.util import repeat_image_tensor

EPSILON = 1e-16
class PhasePlateWithReflection(OpticalLayer):
    def __init__(self, shape, scale = 2, reflection_strength_trainable = False, amplitude_trainable = [False, False], phase_trainable = [True, True], initialization = 'zeros', **kwargs):
        '''
        A simple layer to multiply the incoming complex field with a phase plate that has (possible trainable) amplitude and phase
        shape: the input shape of the image
        scale: upscale the plate
        amplitude_trainable: weather the amplitude of the phaseplate is trainable
        phase_trainable: makes the phase of the phaseplate trainable (enabled by default)
        initialization: The initialization parameter given the the phase part of the phaseplate. defaults to zero, since random values gives essentially a diffuser and give no good gradient
        '''
        super(PhasePlateWithReflection, self).__init__(**kwargs)
        #if np.array(plate_size).size == 1 :
        #    self.N = np.array([plate_size,plate_size])
        #else:
         #   self.N = np.array(plate_size)
        self.scale = scale
        self.plate_shape = shape
        self.amplitude_trainable= amplitude_trainable
        self.phase_trainable = phase_trainable

        self.phaseplate_shape = [1, shape[0], shape[1],1]
        if initialization =='zeros':
            init = np.zeros(self.phaseplate_shape)
        elif initialization == 'uniform':
            init = np.random.uniform(0.0, 2*np.pi/20, size = self.phaseplate_shape)
        self.amplitudes = tf.Variable(np.ones(self.phaseplate_shape), trainable = amplitude_trainable, dtype = tf.float32)
        self.phases = tf.Variable(init, trainable= phase_trainable,dtype = tf.float32)

        self.amplitudes_reflection = tf.Variable(np.ones(self.phaseplate_shape), trainable = amplitude_trainable, dtype = tf.float32)
        self.phases_reflection = tf.Variable(init, trainable= phase_trainable,dtype = tf.float32)

        self.reflection_strength = self.add_weight( 0.1, name = 'reflection_strength',trainable = reflection_strength_trainable)


    def call(self, Input):
        assert( Input.shape[1] >= self.phases.shape[1]*self.scale)
        assert( Input.shape[2] >= self.phases.shape[2]*self.scale)
        #assert( Input.shape[3] == self.phases.shape[3])
        diffx = Input.shape[1] - self.phases.shape[1]*self.scale
        diffy = Input.shape[2] - self.phases.shape[2]*self.scale
        #diff_upx = tf.math.round(diffx/2)
        diff_upx= tf.cast(tf.math.round(diffx/2),dtype = tf.int32)
        diff_upy = tf.cast(tf.math.round(diffy/2), dtype = tf.int32)
        diff_downx= tf.cast(tf.math.round(diffx/2 ), dtype = tf.int32)
        diff_downy = tf.cast(tf.math.round(diffy/2),dtype = tf.int32)

        amp = tf.pad(repeat_image_tensor(self.amplitudes,self.scale), [ [0,0],[diff_upx, diff_downx], [diff_upy, diff_downy], [0,0]], "CONSTANT",1.0)

        #amp = tf.image.resize(self.amplitudes, (Input.shape[1], Input.shape[2]), method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        phase = tf.pad(repeat_image_tensor(self.phases, self.scale), [ [0,0],[diff_upx, diff_downx], [diff_upy, diff_downy], [0,0]], "CONSTANT",0.0)
        #phase = tf.image.resize(self.phases, (Input.shape[1], Input.shape[2]), method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        phase_plate = tf.complex(amp + EPSILON, 0.0) * tf.cast(
            tf.exp(1j * tf.cast(phase, dtype=tf.complex64)), dtype=tf.complex64)
        

        amp_ref = tf.pad(repeat_image_tensor(self.amplitudes_reflection,self.scale), [ [0,0],[diff_upx, diff_downx], [diff_upy, diff_downy], [0,0]], "CONSTANT",1.0)
        phase_ref = tf.pad(repeat_image_tensor(self.phases, self.scale), [ [0,0],[diff_upx, diff_downx], [diff_upy, diff_downy], [0,0]], "CONSTANT",0.0)
        
        phase_plate_ref = tf.complex(amp_ref + EPSILON, 0.0) * tf.cast(
            tf.exp(1j * tf.cast(phase_ref, dtype=tf.complex64)), dtype=tf.complex64)

        #print(Input.shape)
        #print(phase_plate.shape)
        new_field = Input * phase_plate (1.0 - self.reflection_strength) + Input * phase_plate_ref * self.reflection_strength


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
            'phase_trainable': self.phase_trainable
        }
        return temp










