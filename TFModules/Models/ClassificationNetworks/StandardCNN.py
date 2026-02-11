import tensorflow as tf


class StandardCNN(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(StandardCNN, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(10, (5,5), activation = 'gelu')
        self.conv2 = tf.keras.layers.Conv2D(30, (5,5), activation = 'gelu')
        self.conv3 = tf.keras.layers.Conv2D(50, (5,5), activation = 'gelu')
        self.conv4 = tf.keras.layers.Conv2D(60, (5,5), activation = 'gelu')

        self.flatten = tf.keras.laysers.Flatten()
        self.dense1 = tf.keras.layers.dense(100)
        self.out = tf.keras.layers.Dense(2)

    def call(self, input):
        conv_out = self.conv4(self.conv3(self.conv2(self.conv1( input))))
        out = self.out(self.dense1(self.flatten(conv_out)))
        return out
        
