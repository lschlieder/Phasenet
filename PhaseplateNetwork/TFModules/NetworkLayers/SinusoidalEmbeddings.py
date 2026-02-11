import tensorflow as tf
from tensorflow.keras.layers import Layer

class SinusoidalEmbeddings(Layer):
    def __init__(self, dim, **kwargs):
        super(SinusoidalEmbeddings, self).__init__(**kwargs)
        self.dim = dim

    def call(self, time):
        #device = time.device
        half_dim = self.dim // 2
        embeddings = tf.math.log(10000.0) / (half_dim - 1)
        embeddings = tf.math.exp(tf.range(half_dim,dtype = tf.float32) * -embeddings)
        embeddings = (time[:, None] * embeddings[None, :])[:,0,:]
        embeddings = tf.concat((tf.math.sin(embeddings), tf.math.cos(embeddings)), axis=-1)
        
        return embeddings