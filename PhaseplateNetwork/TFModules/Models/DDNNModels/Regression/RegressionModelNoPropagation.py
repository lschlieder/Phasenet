import tensorflow as tf
import numpy as np
from PhaseplateNetwork.utils.Constraints.MinMaxConstraint import MinMaxConstraint

def input_encoding(input):
    #print('input encoding')
    phase_input = tf.complex(1.0, 0.0) * tf.math.exp(tf.complex(0.0, input*2*np.pi))
    amplitude_input = tf.complex(input, 0.0)
    constant_input = tf.complex(tf.ones_like(input),0.0)
    return phase_input, amplitude_input, constant_input

def input_encoding_phase(input):
    phase_input = tf.complex(1.0, 0.0) * tf.math.exp(tf.complex(0.0, input*2*np.pi))
    amplitude_input = tf.complex(input, 0.0)
    constant_input = tf.complex(tf.ones_like(input),0.0)
    return phase_input, amplitude_input, constant_input

class RegressionModelNoPropagation(tf.keras.Model):
    def __init__(self, num_outputs:int = 100, input_enc = 'both', regularizer = None, mode='negative', const_min = 0.0, const_max = 1.0, **kwargs):
        super(RegressionModelNoPropagation, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        num_outputs = int(num_outputs)
        self.L = 1
        self.mode = mode
        self.input_enc = input_enc
        self.num_encoding_dims = 3 if input_enc == 'both' else 2

        #constraint = tf.keras.constraints.MinMaxNorm(min_value = 0.0, max_value = 1.0)
        #self.real_mat = self.add_weight(shape = (num_outputs,3), constraint = constraint)
        #self.imag_mat = self.add_weight(shape = (num_outputs,3), constraint = constraint)
        
        constraint = MinMaxConstraint(min = const_min, max = const_max)

        def my_init(shape, val_max = 0.001, dtype = tf.complex64):
            real = tf.random.uniform(shape, -val_max, val_max)
            imag = tf.random.uniform(shape, -val_max, val_max)
            return tf.complex(real,imag)
        
        init = lambda shape, dtype: my_init(shape, val_max = 10/(6*self.num_outputs), dtype = dtype)
        
        self.mat = self.add_weight(name = "matrix_mul", shape = (num_outputs, self.num_encoding_dims),initializer = init, dtype = tf.complex64)

        if mode == 'negative':
            #print('mode is negative')
            self.neg_mat = self.add_weight(name = "neg_mat", shape = (num_outputs, self.num_encoding_dims),initializer = init, dtype = tf.complex64)
        


    def call(self, input):

        input = tf.cast(input, tf.float32)
        phase, amp, const = input_encoding(input)
        if self.input_enc == 'both':
            inp = tf.reshape(tf.concat((phase,amp,const), axis = 1) ,(-1,3,1))
        elif self.input_enc == 'phase':
            inp = tf.reshape(tf.concat( (phase, const), axis = 1), (-1, 2,1))
        elif self.input_enc == 'amp':
            inp = tf.reshape(tf.concat( (amp, const), axis = 1), (-1, 2,1))
        #mat = tf.complex(self.real_mat, self.imag_mat)
        

        
        intermediate = tf.squeeze(tf.matmul(self.mat, inp))
        out = tf.abs(intermediate)**2



        if self.num_outputs ==1:
            out = tf.reshape(out, (-1, 1))
        elif input.shape[0] == 1:
            out = tf.reshape(out, (1,-1))


        sum = tf.reduce_sum(out, axis = 1)

        if self.mode == 'negative':
            #print('mode is negative')
            intermediate_neg = tf.squeeze(tf.matmul(self.neg_mat, inp))
            out_neg = tf.abs(intermediate_neg)**2
            sum_neg = tf.reduce_sum(out_neg, axis = 1)
            sum = sum - sum_neg
        #sum_neg = tf.reduce_sum(out_neg, axis = 1)
        #sum = sum - sum_neg
        return sum
    
    def get_input_shape(self):
        return [None, 1]
    
    def get_input_size(self):
        return [None, 1]
    
    def get_intermediate_functions(self, input):

        input = tf.cast(input, tf.float32)
        phase, amp, const = input_encoding(input)
        if self.input_enc == 'both':
            inp = tf.reshape(tf.concat((phase,amp,const), axis = 1) ,(-1,3,1))
        elif self.input_enc == 'phase':
            inp = tf.reshape(tf.concat( (phase, const), axis = 1), (-1, 2,1))
        elif self.input_enc == 'amp':
            inp = tf.reshape(tf.concat( (amp, const), axis = 1), (-1, 2,1))


        out = tf.squeeze(tf.matmul(self.mat, inp))
        if self.num_outputs ==1:
            out = tf.reshape(out, (-1, 1))
        elif input.shape[0] == 1:
            out = tf.reshape(out, (1,-1))      
        #print(self.mat.shape)
        #print(inp.shape)
        #print(out.shape)
        #out = tf.abs(out)**2
        return out