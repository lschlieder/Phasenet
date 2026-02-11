import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
from PhaseplateNetwork.utils.util import repeat_image_tensor

EPSILON = 1e-16
class PhasePlateVariableInput(OpticalLayer):
    def __init__(self, shape, scale = 2, **kwargs):
        '''
        A simple layer to multiply the incoming complex field with a phase plate that has (possible trainable) amplitude and phase
        shape: the input shape of the image
        scale: upscale the plate
        '''
        super(PhasePlateVariableInput, self).__init__(**kwargs)
        #if np.array(plate_size).size == 1 :
        #    self.N = np.array([plate_size,plate_size])
        #else:
         #   self.N = np.array(plate_size)
        self.scale = scale
        self.plate_shape = shape
        #self.amplitude_trainable= amplitude_trainable
        #self.phase_trainable = phase_trainable

        self.phaseplate_shape = [1, shape[0], shape[1],1]
        #self.amplitudes = tf.Variable(np.ones(self.phaseplate_shape), trainable = amplitude_trainable, dtype = tf.float32)
        #self.phases = tf.Variable(np.zeros(self.phaseplate_shape), trainable= phase_trainable,dtype = tf.float32)



    def call(self, Input, Phaseplate):
        assert( Input.shape[1] >= self.phaseplate_shape[1]*self.scale)
        assert( Input.shape[2] >= self.phaseplate_shape[2]*self.scale)

        assert( Phaseplate.shape[1] >= self.phaseplate_shape[1] * self.scale)
        assert( Phaseplate.shape[2] >= self.phaseplate_shape[2] * self.scale)
        #assert( Input.shape[3] == self.phases.shape[3])
        diffx = Input.shape[1] - self.phaseplate_shape[1]*self.scale
        diffy = Input.shape[2] - self.phaseplate_shape[2]*self.scale
        #diff_upx = tf.math.round(diffx/2)
        diff_upx= tf.cast(tf.math.round(diffx/2),dtype = tf.int32)
        diff_upy = tf.cast(tf.math.round(diffy/2), dtype = tf.int32)
        diff_downx= tf.cast(tf.math.round(diffx/2 ), dtype = tf.int32)
        diff_downy = tf.cast(tf.math.round(diffy/2),dtype = tf.int32)

        #amp = tf.pad(repeat_image_tensor(self.amplitudes,self.scale), [ [0,0],[diff_upx, diff_downx], [diff_upy, diff_downy], [0,0]], "CONSTANT",1.0)

        #phase = tf.pad(repeat_image_tensor(self.phases, self.scale), [ [0,0],[diff_upx, diff_downx], [diff_upy, diff_downy], [0,0]], "CONSTANT",0.0)
        #phase_plate = tf.complex(amp + EPSILON, 0.0) * tf.cast(
        #    tf.exp(1j * tf.cast(phase, dtype=tf.complex64)), dtype=tf.complex64)
        phaseplate_padded = tf.pad(repeat_image_tensor(Phaseplate, self.scale), [ [0,0],[diff_upx, diff_downx], [diff_upy, diff_downy], [0,0]], "CONSTANT",1.0+0.0*1j)

        #print(Input.shape)
        #print(phase_plate.shape)
        new_field = Input * phaseplate_padded
        return new_field

    def compute_output_shape(self, input_shape):
        #assert( input_shape[1] == self.phases.shape[1])
        #assert( input_shape[2] == self.phases.shape[2])
        #assert( input_shape[3] == self.phases.shape[3])
        output_shape = []
        for i in range(0,len(input_shape)):
            output_shape.append(input_shape[i])
        return output_shape


    def get_config(self):
        #shape, scale = 2, amplitude_trainable = False, phase_trainable = True
        temp = {
            'shape': self.plate_shape,
            'scale': self.scale
        }
        return temp