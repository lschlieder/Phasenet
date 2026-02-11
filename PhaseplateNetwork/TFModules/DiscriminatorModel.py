import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PhaseplateNetwork.TFModules.PropagationLayer import PropagationLayer

class DiscriminatorModel(tf.keras.Model):
    def __init__(self):
        super(DiscriminatorModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (5,5), strides = (2,2), activation = tf.keras.layers.LeakyReLU())
        #self.max_pool1 = tf.keras.layers.MaxPool2D(pool_size = (2,2))
        self.conv2 = tf.keras.layers.Conv2D(128, (5,5),strides = (2,2), activation = tf.keras.layers.LeakyReLU())
        #self.max_pool2 = tf.keras.layers.MaxPool2D(pool_size = (2,2))
        self.conv3 = tf.keras.layers.Conv2D(40, (5,5), strides = (2,2), activation = tf.keras.layers.LeakyReLU())
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(100, activation = tf.keras.layers.LeakyReLU())
        self.dense_out = tf.keras.layers.Dense(1)

    def call(self, inputs, **kwargs):
        #print(inputs.shape)
        #print(inputs.dtype)
        #res = self.dense_out(self.dense1( self.flatten( self.conv3( self.max_pool2(self.conv2( self.max_pool1(self.conv1(inputs))))))))
        res = self.dense_out(self.dense1( self.flatten( self.conv3( self.conv2( self.conv1(inputs))))))
        #res = self.dense_out(self.dense1(self.flatten(inputs)))
        return res
