import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt


class gradient_descent_block(tf.keras.layers.Layer):

    def __init__(self,*kwargs):
        super(gradient_descent_block, self).__init__(*kwargs)


    def build(self, input_shape):
        self.learning_rate  = tf.Variable(1000*tf.ones([1]), trainable=True)
        self.loss_learning_rate = tf.Variable(tf.zeros([1]), trainable=True)

    def call(self, input):
        #print(input.shape)
        #print(self.learning_rate.shape)
        #print( (self.learning_rate * input[:,:,:,1:2]))
        #print(self.learning_rate)
        output_phase = input[:,:,:,0:1] - self.learning_rate*input[:,:,:,1:2]#/(self.loss_learning_rate*tf.reduce_mean(input[:,:,:,2], axis = [1,2]))
        #print(self.learning_rate)
        #plt.imshow(input[0,:,:,1])
        #plt.figure()
        #plt.imshow(output_phase[0,:,:,0])
        #plt.show()
        return output_phase

    def clear_state(self):
        #shape = self.internal_state.shape
        #self.internal_state.assign(tf.zeros([shape[0],shape[1], shape[2], 1]))
        return

class RecurrentBlock(tf.keras.layers.Layer):
    def __init__(self,*kwargs):
        super(RecurrentBlock, self).__init__(*kwargs)

        #self.internal_state = tf.Variable(tf.zeros([image_size//8*image_size//8]))
        self.conv_down_11 = tf.keras.layers.Conv2D(6, 6, padding = 'same', activation = 'relu')
        self.conv_down_12 = tf.keras.layers.Conv2D(6, 6, padding = 'same', activation = 'relu')
        self.downsampling_1 = tf.keras.layers.MaxPool2D((2,2))
        self.conv_down_21 = tf.keras.layers.Conv2D(12, 6, padding = 'same', activation = 'relu')
        self.conv_down_22 = tf.keras.layers.Conv2D(12,6, padding = 'same', activation = 'relu')
        self.downsampling_2 = tf.keras.layers.MaxPool2D((2,2))
        self.conv_down_31 = tf.keras.layers.Conv2D(24, 6, padding= 'same', activation = 'relu')
        self.conv_down_32 = tf.keras.layers.Conv2D(24,6, padding = 'same', activation = 'relu')
        self.downsampling_3 = tf.keras.layers.MaxPool2D((2,2))


        self.mashed_signal_conv = tf.keras.layers.Conv2D(48, 6, padding = 'same', activation = 'relu')
        self.new_signal_conv = tf.keras.layers.Conv2D(1, 6, padding = 'same', activation = 'relu')

        self.conv_up1 = tf.keras.layers.Conv2DTranspose(24, 6, 2, padding = 'same')
        self.conv_up2 = tf.keras.layers.Conv2DTranspose(24,6,2, padding = 'same')
        self.conv_up3 = tf.keras.layers.Conv2DTranspose(24, 6, 2, padding = 'same')
        self.output_phase_conv = tf.keras.layers.Conv2D(1, 6, padding = 'same', activation = 'tanh')

    def build(self, input_shape):
        self.internal_state = tf.Variable(tf.zeros([input_shape[0],input_shape[1]//8, input_shape[2]//8, 1]), trainable=False)

    def call(self, input):
        down_1 = self.downsampling_1(self.conv_down_12(self.conv_down_11(input)))
        down_2 = self.downsampling_2(self.conv_down_22(self.conv_down_21(down_1)))
        down_3 = self.downsampling_3(self.conv_down_32(self.conv_down_31(down_2)))

        concat = tf.concat([down_3, self.internal_state],axis = 3)
        mashed_signal = self.mashed_signal_conv(concat)

        new_state = self.new_signal_conv(mashed_signal)

        output_phase = input[:,:,:,0:1] + self.output_phase_conv(self.conv_up3( self.conv_up2( self.conv_up1(mashed_signal))))
        self.internal_state.assign(new_state)
        return output_phase

    def clear_state(self):
        shape = self.internal_state.shape
        self.internal_state.assign(tf.zeros([shape[0],shape[1], shape[2], 1]))



class LSTMO( tf.keras.Model):
    def __init__(self, loss_function, iterations = 70, **kwargs):
        super(LSTMO,self).__init__(**kwargs)

        self.iterations = iterations
        self.recurrent_block = RecurrentBlock()
        #self.recurrent_block = gradient_descent_block()
        self.loss_function = loss_function
        self.x_arr = []


    def build(self, input_shape):
        self.recurrent_block.build([input_shape[0],input_shape[1],input_shape[2],3])
        #self.internal_state = tf.Variable(tf.zeros([input_shape[0],input_shape[1]//8, input_shape[2]//8, 1]), trainable=False)

    def call(self, input):
        #print(input.shape)
        x = tf.random.uniform( shape = [input.shape[0], input.shape[1], input.shape[2], 1])
        x_array = []
        loss_array = []
        self.recurrent_block.clear_state()
        for i in range(0,self.iterations):

            with tf.GradientTape(persistent = True) as g:
                g.watch(x)
                loss = self.loss_function(x, input)
            gradient = []
            #print(loss[0])
            loss_array.append(loss)

            gradient = tf.stop_gradient(g.gradient(loss,x))
            del g
            #gradient = tf.stop_gradient(tf.stack(gradient))
            #plt.imshow(gradient[0,:,:,0])
            #plt.figure()
            #plt.imshow(x[0,:,:,0])
            #plt.show()

            loss_tiled = tf.tile(tf.reshape(loss, [ loss.shape[0],1,1,1]), [1, x.shape[1], x.shape[2], 1])
            #print(gradient.shape)
            block_inp = tf.concat( [x,gradient,loss_tiled], axis = 3)
            x = self.recurrent_block.call(block_inp)
            x_array.append(tf.identity(x))
            #print(x.shape)
        self.add_loss(tf.reduce_mean(tf.concat(loss_array,axis = 0 )))
        self.x_arr = x_array
        return x

    def get_image_variables(self):

        return []







