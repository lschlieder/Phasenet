import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
import matplotlib.pyplot as plt

class TrainableLensePhasePlate(OpticalLayer):

    def __init__(self, shape, focus_number = 50, **kwargs):
        super(LensePhasePlate, self).__init__(**kwargs)
        self.phaseplate_shape = shape
        self.focus_number = focus_number

        starting_positions = tf.random.uniform([focus_number, 2],-1,1)
        starting_size = tf.tile([[1.0,1.0]], [focus_number,1])
        #print(starting_size.shape)
        #print(starting_positions.shape)
        #parameters = tf.concat([starting_positions, starting_size], axis = 1)
        self.positions = tf.Variable(starting_positions,trainable = True)
        self.size = tf.Variable(starting_size, trainable= True)
        #parameters = tf.tile( []
        #self.parameters = tf.Variable( parameters, dtype = tf.float32)
        self.amplitudes = tf.reshape(tf.constant(np.ones(self.phaseplate_shape), dtype = tf.float32), [1, shape[0], shape[1], 1])

        x = np.linspace(-1, 1, shape[0])
        y = -np.linspace(-1, 1, shape[1])
        x, y = np.meshgrid(x, y)
        self.x = x
        self.y = y

    def call(self, Input):

        #phases = tf.zeros([Input.shape[1], Input.shape[2]],dtype = tf.float32)
        lense_arr = []
        for i in range(0, self.focus_number):
            #print(self.x.shape)
            #print(self.parameters.shape)
            plate = self.size[i,0]*(1-tf.math.exp(-(self.x-self.positions[i,0])**2 )) + self.size[i,1]*(1-tf.math.exp(-(self.y- self.positions[i,1])**2))
            lense_arr.append(plate)

        phases = tf.math.reduce_sum( tf.stack(lense_arr), axis = 0)


        phases = tf.reshape(phases, [1,Input.shape[1],Input.shape[2], 1])

        phase_plate = tf.complex(self.amplitudes, 0.0) * tf.cast(
            tf.exp(1j * tf.cast(phases, dtype=tf.complex64)), dtype=tf.complex64)

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
        phases = tf.zeros([self.phaseplate_shape[0], self.phaseplate_shape[1]],dtype = tf.float32)

        lense_arr = []
        #print(self.focus_number)
        #print(self.size.shape)

        for i in range(0, self.focus_number):
            #print(self.x.shape)
            #print(self.parameters.shape)

            #phase = tf.math.minimum(self.size[i,0]*(self.x-self.positions[i,0])**2 + self.size[i,1]*(self.y- self.positions[i,1])**2, 2*np.pi)
            phase = self.size[i,0]*(1-tf.math.exp(-(self.x-self.positions[i,0])**2 )) + self.size[i,1]*(1-tf.math.exp(-(self.y- self.positions[i,1])**2))
            plt.imshow(phase)
            plt.colorbar()
            plt.show()
            #print(phase.shape)
            lense_arr.append(phase)
        #print(self.trainable_variables)
        #print(self.positions)
        #print(self.size)
        #print(lense_arr)
        #print(lense_arr)
        #print('length lens arr: {}'.format(len(lense_arr)))
        phases = tf.math.reduce_sum( tf.stack(lense_arr), axis = 0)% (np.pi*2)
        #print(phases)
        return phases