import tensorflow as tf
import PhaseplateNetwork.TFModules.OpticalLayers.Encodings as EN

class LinearComplexModel(tf.keras.Model):
    def __init__(self, batch_num:int, pixel:int, **kwargs):
        super(LinearComplexModel,self).__init__(**kwargs)

        self.encoding = EN.IntensityEncoding()
        self.batch = int(batch_num)
        self.x,self.y = int(pixel), int(pixel)
        dense_out = self.x*self.y
        self.linear_layer = tf.keras.layers.Dense(dense_out)
        self.flatten = tf.keras.layers.Flatten()


    def call(self, input):
        enc = self.encoding(input)
        out = self.flatten(enc)
        out = tf.abs(self.linear_layer(out))
        out = self.flatten(out)
        return tf.reshape(out, shape = (self.batch, self.x,self.y,1))
    
    def get_input_shape(self):
        return (self.batch, self.x, self.y, 1)
    
    def get_image_variables(self):
        return []
    