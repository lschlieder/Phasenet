import tensorflow as tf


class conv_layer(tf.keras.layers.Layer):
    def __init__(self,out_blocks = 8, middle_blocks = 4, activation = 'relu', **kwargs):
        super(conv_layer,self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(middle_blocks, (5,5), activation = 'relu', padding = 'same')
        self.conv2 = tf.keras.layers.Conv2D(out_blocks, (5,5), padding = 'same', activation = activation)
        
        
    def call(self, input):
        return self.conv2(self.conv1(input))
    
    
class up_conv_layer(tf.keras.layers.Layer):
    def __init__(self,out_blocks = 8, **kwargs):
        super(up_conv_layer,self).__init__(**kwargs)
        
        self.conv1 = tf.keras.layers.Conv2DTranspose(out_blocks*2, (5,5), activation = 'relu', padding = 'same')
        self.conv2 = tf.keras.layers.Conv2DTranspose(out_blocks, (5,5),strides = (2,2),  activation = 'relu',padding = 'same')
        
    def call(self,input):
        return self.conv2(self.conv1(input))
    

class unet(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(unet, self).__init__(**kwargs)
        
        self.layer1 = conv_layer(out_blocks = 32, middle_blocks = 1)
        self.down_1 = tf.keras.layers.MaxPool2D(padding = 'same')
        
        self.layer2 = conv_layer(out_blocks = 64, middle_blocks = 2)
        self.down_2 = tf.keras.layers.MaxPool2D(padding = 'same')
        
        #self.layer3 = conv_layer(out_blocks = 8, middle_blocks = 4)
        #self.down_3 = tf.keras.layers.MaxPool2D(padding = 'same')
        
        
        #self.up_3 = up_conv_layer(out_blocks=4)
        self.up_2 = up_conv_layer(out_blocks=32)
        self.up_1 = up_conv_layer(out_blocks=16)
        
        
        self.t_encoding = tf.keras.layers.Dense(28*28)
        
        self.last_conv = conv_layer(out_blocks = 1, middle_blocks = 10, activation = 'linear')
        
    def call(self, input):
        img, t = input
        
        t_enc = tf.reshape( self.t_encoding(t), (-1, 28,28,1))
        
        in1 = tf.concat((img, t_enc), axis = 3)
        
        d1 = self.down_1(self.layer1(in1))
        d2 = self.down_2(self.layer2(d1))
        #d3 = self.down_3(self.layer3(d2))
        
        #u3 = self.up_3(d3)
        u2 = self.up_2(d2)
        u1 = self.up_1( tf.concat((u2, d1), axis = 3) )
        
        out = self.last_conv(tf.concat( (u1,in1), axis = 3) )
        
        
        return out