import tensorflow as tf


class ConvBlock(tf.keras.Model):
    def __init__(self, out= 4, out_activation = None ):
        '''
        Simple Convolution block for usage in ResNetGBNet
        :param out: output channels
        :param out_activation: keras activation string for the last layer
        '''
        super(ConvBlock,self).__init__()
        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()
        self.batch_normalization3 = tf.keras.layers.BatchNormalization()
        self.batch_normalization4 = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.Conv2D(50, 8, padding = 'same', activation = 'relu')
        self.conv2 = tf.keras.layers.Conv2D(50, 8 , padding = 'same', activation = 'relu')
        self.conv3 = tf.keras.layers.Conv2D(50, 8, padding = 'same', activation = 'relu')
        self.conv4 = tf.keras.layers.Conv2D(50, 8, padding = 'same', activation = 'relu')
        self.conv_out = tf.keras.layers.Conv2D(out, 20, padding = 'same',activation = out_activation)

    def call(self, inputs):
        '''
        Calculates the output of the block
        .params: inputs: Input imagaes of shape [batch, x,y, channels]
        '''
        out = self.conv_out(self.conv4(self.batch_normalization4(self.conv3(self.batch_normalization3(self.conv2(self.batch_normalization2(self.conv1(self.batch_normalization1(inputs)))))))))
        return out
