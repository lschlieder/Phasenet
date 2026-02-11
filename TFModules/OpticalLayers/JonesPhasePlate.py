import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
from PhaseplateNetwork.utils.util import repeat_image_tensor
import matplotlib.pyplot as plt

EPSILON = 1e-16
class JonesPhasePlate(OpticalLayer):
    def __init__(self, shape, scale = 2, amplitude_trainable = False, phase_trainable = True, action = 'x', **kwargs):
        '''
        A simple layer to multiply the incoming complex field with a phase plate that has (possible trainable) amplitude and phase
        shape: the input shape of the image
        scale: upscale the plate
        '''
        super(JonesPhasePlate, self).__init__(**kwargs)
        #if np.array(plate_size).size == 1 :
        #    self.N = np.array([plate_size,plate_size])
        #else:
         #   self.N = np.array(plate_size)
        self.scale = scale
        self.plate_shape = shape
        self.amplitude_trainable= amplitude_trainable
        self.phase_trainable = phase_trainable
        self.action = action

        if action == 'x':
            self.phaseplate_shape = [1, shape[0], shape[1],1]
            self.amplitudes_x = tf.Variable(np.ones(self.phaseplate_shape),trainable = amplitude_trainable, dtype = tf.float32)
            self.amplitudes_y = tf.Variable(np.ones(self.phaseplate_shape),trainable = False, dtype = tf.float32)
            #self.amplitudes = tf.Variable(tf.concat((self.amplitudes_x, self.amplitudes_y), axis = 3))
            self.phases_x = tf.Variable(np.zeros(self.phaseplate_shape), trainable= phase_trainable,dtype = tf.float32)
            self.phases_y = tf.Variable(np.zeros(self.phaseplate_shape), trainable = False, dtype = tf.float32)
            #self.phases = tf.Variable(tf.concat((self.phases_x, self.phases_y), axis = 3))
        elif action == 'y':
            self.phaseplate_shape = [1, shape[0], shape[1],1]
            self.amplitudes_x = tf.Variable(np.ones(self.phaseplate_shape),trainable = False, dtype = tf.float32)
            self.amplitudes_y = tf.Variable(np.ones(self.phaseplate_shape),trainable = amplitude_trainable, dtype = tf.float32)
            #self.amplitudes = tf.Variable(tf.concat((self.amplitudes_x, self.amplitudes_y), axis = 3))
            self.phases_x = tf.Variable(np.zeros(self.phaseplate_shape), trainable= False,dtype = tf.float32)
            self.phases_y = tf.Variable(np.zeros(self.phaseplate_shape), trainable = phase_trainable, dtype = tf.float32)
            #self.phases = tf.Variable(tf.concat((self.phases_x, self.phases_y), axis = 3))
        elif action =='xy':
            self.phaseplate_shape = [1, shape[0], shape[1],1]
            self.amplitudes_x = tf.Variable(np.ones(self.phaseplate_shape),trainable = amplitude_trainable, dtype = tf.float32)
            self.amplitudes_y = tf.Variable(np.ones(self.phaseplate_shape),trainable = amplitude_trainable, dtype = tf.float32)
            #self.amplitudes = tf.Variable(tf.concat((self.amplitudes_x, self.amplitudes_y), axis = 3))
            self.phases_x = tf.Variable(np.zeros(self.phaseplate_shape), trainable= phase_trainable,dtype = tf.float32)
            self.phases_y = tf.Variable(np.zeros(self.phaseplate_shape), trainable = phase_trainable, dtype = tf.float32)
            #self.phases = tf.Variable(tf.concat((self.phases_x, self.phases_y), axis = 3))
        else:
            raise SystemExit('action not known. Action: {}'.format(action))

        self.amplitudes = tf.concat((self.amplitudes_x, self.amplitudes_y), axis = 3)
        self.phases = tf.concat((self.phases_x, self.phases_y), axis = 3)
        self.phaseplate_shape = [1, shape[0], shape[1],2]


    def get_image_variables(self):
        return {'amplitudes_x': self.amplitudes_x, 
                'amplitudes_y': self.amplitudes_y,
                'phases_x':  self.phases_x,
                'phases_y': self.phases_y}

    def call(self, Input, **kwargs):
        assert( Input.shape[1] >= self.phases.shape[1]*self.scale)
        assert( Input.shape[2] >= self.phases.shape[2]*self.scale)
        assert( Input.shape[3] == 2)
        #assert( Input.shape[3] == self.phases.shape[3])

        self.amplitudes = tf.concat((self.amplitudes_x, self.amplitudes_y), axis = 3)
        self.phases = tf.concat((self.phases_x, self.phases_y), axis = 3)

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

        #print(Input.shape)
        #print(phase_plate.shape)
        new_field = Input * phase_plate
        #print(phase_plate.shape)
        return new_field

    def compute_output_shape(self, input_shape):
        #assert( input_shape[1] == self.phases.shape[1])
        #assert( input_shape[2] == self.phases.shape[2])
        #assert( input_shape[3] == self.phases.shape[3])
        output_shape = []
        for i in range(0,len(input_shape)):
            output_shape.append(input_shape[i])
        output_shape[3] = 2
        return output_shape

    def get_image_variables(self):
        if len(self.trainable_variables) != 0:
            #print(self.trainable_variables[0])
            #print(len(self.trainable_variables))
            #self.amplitudes_x = tf.Variable(np.ones(self.phaseplate_shape),trainable = amplitude_trainable, dtype = tf.float32)
            #self.amplitudes_y = tf.Variable(np.ones(self.phaseplate_shape),trainable = False, dtype = tf.float32)
            #self.amplitudes = tf.Variable(tf.concat((self.amplitudes_x, self.amplitudes_y), axis = 3))
            #self.phases_x = tf.Variable(np.zeros(self.phaseplate_shape), trainable= phase_trainable,dtype = tf.float32)
            #self.phases_y = tf.Variable(np.zeros(self.phaseplate_shape), trainable = False, dtype = tf.float32)

            res = tf.concat(self.trainable_variables, axis = 0)

            #res = res[0,:,:,:]
            #res = tf.transpose(res, [2,0,1])
            #res_x = res[:,:,:,0]
            #res_y = res[:,:,:,1]
            #print(res.shape)

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
            'action': self.action
        }
        return temp










