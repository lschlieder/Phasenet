import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
from PhaseplateNetwork.utils.util import repeat_image_tensor, repeat_2d_tensor, repeat

class LCD_Display(OpticalLayer):
    '''
    LCD_Display for Jones calculus images.
    Same principle as the phase plate layer with scaling etc.
    '''
    def __init__(self, shape, scale = 2, trainable = True, alpha = (np.pi/2)/(5*10**-4),input_angle = 0.0,  beta = 700000, depth = 5*10**-4,**kwargs):
        '''
        A layer to multiply the incoming jones matrix field with a LCD display matrix ( phase delay plate followed by rotation matrix).
        alpha and beta parameters depending on voltage given to LCD pixel. Linear interpolation assumed.
        For more information see:
        Introduction to fourier optics, Joseph W. Goodman, P. 191
        Fundamentals of photonics, BAHAA E. A. SALEH, MALVIN CARL TEICH, P. 230
        shape: the input shape of the image
        scale: upscale the plate
        '''
        super(LCD_Display, self).__init__(**kwargs)
        self.scale = scale
        self.trainable= trainable
        self.amplitudes_shape = [1, shape[0], shape[1],1]
        self.amplitudes = tf.Variable(np.zeros(self.amplitudes_shape, 'float32'), trainable = trainable, dtype = tf.float32, constraint = lambda t:tf.clip_by_value(t,0.0,1.0))
        self.alpha = alpha
        self.input_angle = input_angle
        self.beta = beta
        self.depth = depth
        self.full_angle = -alpha * depth
        self.retardation = beta * depth


    def get_rotation_matrix(self, angle):
        rotation_mat_up = tf.concat((tf.math.cos(angle), -tf.math.sin(angle)), axis=3)
        rotation_mat_down = tf.concat((tf.math.sin(angle), tf.math.cos(angle)), axis=3)
        rotation_mat = tf.stack((rotation_mat_up, rotation_mat_down), axis=4)
        return rotation_mat

    def get_retardation_matrix(self, retardation):
        retardation_mat_up = tf.concat((tf.ones_like(retardation), tf.zeros_like(retardation)), axis=3)
        retardation_mat_down = tf.concat((tf.zeros_like(retardation), tf.math.exp(1j*retardation)), axis=3)

        retardation_mat = tf.stack((retardation_mat_up, retardation_mat_down), axis = 4)
        return retardation_mat

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[2],input_shape[3]]

    def call(self,input):

        def get_propagation_masks(plate_shape, input_shape):
            temp = tf.zeros(plate_shape, dtype = tf.complex64)
            diffx = input_shape[1] - plate_shape[1]
            diffy = input_shape[2] - plate_shape[2]
            diff_upx = tf.cast(tf.math.round(diffx / 2), dtype=tf.int32)
            diff_upy = tf.cast(tf.math.round(diffy / 2), dtype=tf.int32)
            diff_downx = tf.cast(tf.math.round(diffx / 2), dtype=tf.int32)
            diff_downy = tf.cast(tf.math.round(diffy / 2), dtype=tf.int32)
            temp = tf.pad(temp, [[0,0], [diff_upx, diff_downx], [diff_upy, diff_downy], [0,0]], "CONSTANT", 1.0)
            inv_mask = -(temp - 1.0)
            return temp, inv_mask

        assert(input.shape[3] == 2)
        assert( input.shape[1] >= self.amplitudes.shape[1]*self.scale)
        assert( input.shape[2] >= self.amplitudes.shape[2]*self.scale)

        diffx = input.shape[1] - self.amplitudes.shape[1]*self.scale
        diffy = input.shape[2] - self.amplitudes.shape[2]*self.scale
        #diff_upx = tf.math.round(diffx/2)
        diff_upx= tf.cast(tf.math.round(diffx/2),dtype = tf.int32)
        diff_upy = tf.cast(tf.math.round(diffy/2), dtype = tf.int32)
        diff_downx= tf.cast(tf.math.round(diffx/2 ), dtype = tf.int32)
        diff_downy = tf.cast(tf.math.round(diffy/2),dtype = tf.int32)

        img_dims_x = [diff_upx, self.amplitudes.shape[1]*self.scale+ diff_upx]
        img_dims_y = [diff_upy, self.amplitudes.shape[2]*self.scale + diff_upy]

        #tf.pad(repeat_image_tensor(self.amplitudes, self.scale),
        #       [[0, 0], [diff_upx, diff_downx], [diff_upy, diff_downy], [0, 0]], "CONSTANT", 1.0)

        #repeated_amplitudes = repeat_2d_tensor(self.amplitudes,self.scale)
        repeated_amplitudes = repeat(self.amplitudes,self.scale, 1)
        repeated_amplitudes = repeat(repeated_amplitudes, self.scale, 2)

        #def repeat(input, repeat=2, dimension=0):
        angle = tf.complex(self.full_angle * (1.0-repeated_amplitudes),0.0)
        retardation = tf.complex(self.retardation *(1.0-repeated_amplitudes),0.0)
        rotation_mat = self.get_rotation_matrix(angle)
        retardation_mat = self.get_retardation_matrix(retardation)

        #### Get only the input in the middle of the propagation area
        cropped_input = input[:,img_dims_x[0]:img_dims_x[1], img_dims_y[0]:img_dims_y[1],:]

        temp = tf.linalg.matvec(retardation_mat, cropped_input)
        temp = tf.linalg.matvec(rotation_mat, temp)
        print(temp.shape)
        temp = tf.pad(temp, [[0,0], [diff_upx, diff_downx],[diff_upy, diff_downy], [0,0]], "CONSTANT", 0.0)

        #get masks

        mask, inv_mask = get_propagation_masks(repeated_amplitudes.shape, input.shape)

        #calculate output with border
        output = mask * input + temp*  inv_mask
        return output

    def get_config(self):
        #shape, scale = 2, amplitude_trainable = False, phase_trainable = True
        #self, shape, scale = 2, trainable = True, alpha = (np.pi / 2) / (
        #            5 * 10 ** -4), input_angle = 0.0, beta = 700000, depth = 5 * 10 ** -4, ** kwargs):

        temp = {
            'shape': self.amplitudes_shape,
            'scale': self.scale,
            'trainable': self.trainable,
            'alpha': self.alpha,
            'input_angle' : self.input_angle,
            'beta' : self.beta,
            'depth' : self.depth
        }
        return temp


    def get_image_variables(self):
        if len(self.trainable_variables) != 0:
            res = tf.concat(self.trainable_variables, axis = 0)
            #print(res.shape)
            res = res[:,:,:,0]
        else:
            res = None
        #print(res.shape)

        return res
