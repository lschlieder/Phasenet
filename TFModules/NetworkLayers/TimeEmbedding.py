import tensorflow as tf

def get_time_embedding(t, max_timesteps=200):
    half_dim = max_timesteps // 2
    #embeddings = math.log(10000) / (half_dim - 1)
    #embeddings = torch.exp(.arange(half_dim) * -embeddings)
    #embeddings = time[:, None] * embeddings[None, :]
    #embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
    embeddings = tf.math.log(10000.0) / (half_dim - 1)
    embeddings = tf.math.exp(tf.range(half_dim,dtype = tf.float32) * -embeddings)
    embeddings = (t[:, None] * embeddings[None, :])[:,0,:]
    embeddings = tf.concat((tf.math.sin(embeddings), tf.math.cos(embeddings)), axis=-1)
    return embeddings

class TimeEmbedding(tf.keras.layers.Layer):
    def __init__(self, img_dim = 32, max_time = 200,**kwargs):
        super(TimeEmbedding,self).__init__(**kwargs)
        self.max_time = max_time
        self.img_dim= img_dim
        self.dense1 = tf.keras.layers.Dense(max_time, activation = 'gelu')
        self.dense2 = tf.keras.layers.Dense(img_dim**2, activation = 'gelu')


    def call(self,input):
        emb = get_time_embedding(input, self.max_time)
        #print(emb.shape)

        return tf.reshape(self.dense2(self.dense1(emb)), (-1, self.img_dim,self.img_dim,1))