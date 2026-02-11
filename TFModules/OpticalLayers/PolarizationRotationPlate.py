import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
from PhaseplateNetwork.utils.util import repeat_image_tensor
import matplotlib.pyplot as plt
from PhaseplateNetwork.utils.util import repeat_image_tensor, repeat_2d_tensor, repeat

class AngleConstraint(tf.keras.constraints.Constraint):
    def __init__(self, min = 0.0, max = np.pi/2):
        self.min = min
        self.max = max

    def __call__(self, w):
        return tf.minimum(tf.maximum(w,self.min), self.max)
    
    def get_config(self):
        return {'min': self.min,
                'max': self.max}
    
EPSILON = 1e-16

class PolarizationRotationPlate(OpticalLayer):
    def __init__(self, shape, scale = 2, max_angle = None, **kwargs):
        '''
        A simple layer to multiply the incoming complex field with a phase plate that has (possible trainable) amplitude and phase
        shape: the input shape of the image
        scale: upscale the plate
        '''
        super(PolarizationRotationPlate, self).__init__(**kwargs)
        #if np.array(plate_size).size == 1 :
        #    self.N = np.array([plate_size,plate_size])
        #else:
         #   self.N = np.array(plate_size)
        self.scale = scale
        self.plate_shape = shape
        self.max_angle = max_angle

        self.phaseplate_shape = [1, shape[0], shape[1],1]
        if max_angle != None:
            print("max_angle:", max_angle)
            #self.max_constraint = tf.keras.constraints.MinMaxNorm(min_value = 0, max_value = max_angle, rate = 1.0)
            self.max_constraint = AngleConstraint(min = 0.0, max = max_angle)
            self.rotation_weights = self.add_weight("rotation_weights", self.phaseplate_shape, tf.float32, initializer = 'zeros', trainable= True, constraint = self.max_constraint)
        else:
            self.rotation_weights = self.add_weight("rotation_weights", self.phaseplate_shape, tf.float32, initializer = 'zeros', trainable=True)
        #self.rotation_weights = tf.Variable(np.zeros(self.phaseplate_shape, 'float32'), trainable = True)


        #self.amplitudes = tf.concat((self.amplitudes_x, self.amplitudes_y), axis = 3)
        #self.phases = tf.concat((self.phases_x, self.phases_y), axis = 3)
        #self.phaseplate_shape = [1, shape[0], shape[1],1]

    '''
    def get_rotation_matrix(self, angle):
        angle = -angle
        rotation_mat_up = tf.concat((tf.math.cos(angle), -tf.math.sin(angle)), axis=3)
        rotation_mat_down = tf.concat((tf.math.sin(angle), tf.math.cos(angle)), axis=3)
        rotation_mat = tf.stack((rotation_mat_up, rotation_mat_down), axis=4)
        return rotation_mat
    '''



    def get_rotation_matrix(self, angle):
        angle = tf.cast(-angle, dtype = tf.float32)
        rotation_mat_up = tf.concat((tf.math.cos(angle), -tf.math.sin(angle)), axis=3)
        rotation_mat_down = tf.concat((tf.math.sin(angle), tf.math.cos(angle)), axis=3)
        rotation_mat = tf.stack((rotation_mat_up, rotation_mat_down), axis=4)
        return tf.cast(rotation_mat, dtype = tf.complex64)

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2],input_shape[3]]



    def call(self, Input, **kwargs):
        assert( Input.shape[1] >= self.rotation_weights.shape[1]*self.scale)
        assert( Input.shape[2] >= self.rotation_weights.shape[2]*self.scale)
        assert( Input.shape[3] == 2)

        def get_propagation_masks(plate_shape, input_shape):
            temp = tf.zeros(plate_shape, dtype=tf.complex64)
            diffx = input_shape[1] - plate_shape[1]
            diffy = input_shape[2] - plate_shape[2]
            diff_upx = tf.cast(tf.math.round(diffx / 2), dtype=tf.int32)
            diff_upy = tf.cast(tf.math.round(diffy / 2), dtype=tf.int32)
            diff_downx = tf.cast(tf.math.round(diffx / 2), dtype=tf.int32)
            diff_downy = tf.cast(tf.math.round(diffy / 2), dtype=tf.int32)
            temp = tf.pad(temp, [[0, 0], [diff_upx, diff_downx], [diff_upy, diff_downy], [0, 0]], "CONSTANT", 1.0)
            inv_mask = -(temp - 1.0)
            return temp, inv_mask
        #assert( Input.shape[3] == self.phases.shape[3])

        #self.amplitudes = tf.concat((self.amplitudes_x, self.amplitudes_y), axis = 3)
        #self.phases = tf.concat((self.phases_x, self.phases_y), axis = 3)

        diffx = Input.shape[1] - self.rotation_weights.shape[1]*self.scale
        diffy = Input.shape[2] - self.rotation_weights.shape[2]*self.scale
        #diff_upx = tf.math.round(diffx/2)
        diff_upx= tf.cast(tf.math.round(diffx/2),dtype = tf.int32)
        diff_upy = tf.cast(tf.math.round(diffy/2), dtype = tf.int32)
        diff_downx= tf.cast(tf.math.round(diffx/2 ), dtype = tf.int32)
        diff_downy = tf.cast(tf.math.round(diffy/2),dtype = tf.int32)


        img_dims_x = [diff_upx, self.rotation_weights.shape[1]*self.scale + diff_upx]
        img_dims_y = [diff_upy, self.rotation_weights.shape[2]*self.scale + diff_upy]

        rot_weights = repeat(repeat(self.rotation_weights,self.scale, 1),self.scale,2)

        angles = tf.complex(rot_weights, 0.0)
        rotation_mat = self.get_rotation_matrix(angles)

        #rotation_weights = tf.pad(repeat_image_tensor(self.rotation_weights, self.scale),[ [0,0],[diff_upx, diff_downx], [diff_upy, diff_downy], [0,0]], "CONSTANT",0.0)

        cropped_input = Input[:,img_dims_x[0]:img_dims_x[1], img_dims_y[0]:img_dims_y[1],:]
        #print(rotation_mat.shape)
        #print(cropped_input.shape)
        temp = tf.linalg.matvec(rotation_mat, cropped_input)
        #temp = tf.linalg.matvec(rotation_mat, temp)
        #print(temp.shape)
        temp = tf.pad(temp, [[0,0], [diff_upx, diff_downx],[diff_upy, diff_downy], [0,0]], "CONSTANT", 0.0)
        mask, inv_mask = get_propagation_masks(angles.shape, Input.shape)
        output = mask * Input + temp*  inv_mask
        return output


    #def get_image_variables(self):
    #    #print(self.trainable_variables())
    #    #print('rotation layer get image variables')
    #    if len(self.trainable_variables) != 0:
    #        res = tf.concat(self.trainable_variables, axis = 0)
    #        #print(self.trainable_variables())
    #        #input()
    #        res = res[0,:,:,:]
    #       res = tf.transpose(res, [2,0,1])
    #
    #    else:
    #        #print('Trainable variables rotation layer:',self.trainable_variables())
    #        res = None
    #    return res
    
    def get_image_variables(self):
        return {'rotation_weights': self.rotation_weights}


    def get_config(self):
        #shape, scale = 2, amplitude_trainable = False, phase_trainable = True
        temp = {
            'shape': self.plate_shape,
            'scale': self.scale,
            'max_angle': self.max_angle
        }
        return temp










