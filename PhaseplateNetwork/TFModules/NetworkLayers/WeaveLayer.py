import tensorflow as tf
import numpy as np

class WeaveLayer(tf.keras.layers.Layer):
    '''
    A custom Keras layer for generating a double-sized image with patches taken from one of the four layers.

    Attributes:
        n (int): Size of the patches.
    Methods:
        __init__(self, n, **kwargs):
            Initializes the layer.

        call(self, input, **kwargs):
            Performs the operation of the layer.

    Example:
        # Define the layer with patch size 3
        weave_layer = WeaveLayer(n=3)
        # Apply the layer to input tensor ( shape: (batch_size, imgy, imgx, 4))
        output = weave_layer(input_tensor)
        #output shape = (batch_size, 2*imgy, 2*imgx, 1)
    '''
    def __init__(self, n, **kwargs):
        super(WeaveLayer, self).__init__(**kwargs)

        def get_kernel(n, channel= 0):
            #mask = np.zeros(4*n**2)
            mask = np.zeros((2*n,2*n))
            image_index = channel//(n**2)
            pos_index = channel%(n**2)

            posx = pos_index%n
            posy = pos_index//n

            posx = posx + n*(image_index in [2,3])
            posy = posy + n*(image_index in [1,3])
            mask[posx, posy] = 1
            return mask
    


        kernel = np.zeros((2*n,2*n, 1,4*n**2))
        for i in range(0, 4*n**2):
            kernel[:,:,0,i] = get_kernel(n, i)
            
        self.kernel = kernel.astype('float32')
        self.n = n
        return
    



    def call(self, input,  **kwargs):
        def stack_image(inp,n ):
            x = inp.shape[2]
            y = inp.shape[1]
            inp = tf.reshape(inp, (-1,y,x,1))
            a = tf.reshape(inp, (-1,y, x//n, n))
            a = tf.transpose(a, (0,2,1,3))
            a = tf.reshape( a, (-1,x//n, y//n, n*n))
            a = tf.transpose(a, (0,2,1,3))
            return a
        
        imgs = input
        
        if imgs.shape[1]%self.n != 0:
            raise ValueError(f"Image cannot be divided into {imgs.shape[1]/self.n} parts (Input shape {imgs.shape}, patch_size: {n})")


        size_x = imgs.shape[2]
        size_y = imgs.shape[1]

        im1 = imgs[:,:,:,0:1]
        im2 = imgs[:,:,:,1:2]
        im3 = imgs[:,:,:,2:3]
        im4 = imgs[:,:,:,3:4]

        im1 = stack_image(im1, self.n)
        im2 = stack_image(im2, self.n)
        im3 = stack_image(im3, self.n)
        im4 = stack_image(im4, self.n)
        
        concat_img = tf.concat((im1, im2, im3, im4), axis = 3)
        u = tf.nn.conv2d_transpose( concat_img, self.kernel, (size_x*2, size_y*2),(self.n*2,self.n*2), padding = 'SAME')
        return u