import tensorflow as tf
import numpy as np

class UnweaveLayer(tf.keras.layers.Layer):
    '''
    A custom Keras layer for reconstructing an image from patches extracted by the WeaveLayer.

    Attributes:
        n (int): Size of the patches.
        size (tuple): Size of the input image.

    Methods:
        __init__(self, n, size, **kwargs):
            Initializes the layer.

        call(self, input, **kwargs):
            Reconstructs the original image from patches of size n x n .

    Example:
        # Define the layer with patch size 3 and output size (64, 64)
        unweave_layer = UnweaveLayer(n=3, size=(64, 64))
        # Apply the layer to input tensor
        output = unweave_layer(input_tensor)
        # output shape ( batch_size, 32, 32, 4)
    '''
        
    def __init__(self, n,size, **kwargs):
        super(UnweaveLayer,self).__init__(**kwargs)
        
        def get_chess_mask(n, size, channel= 0 ):
            ones = np.ones((n,n))
            zeros = np.zeros((n,n))
            if channel ==0:
                top = np.concatenate((ones,zeros), axis = 0)
                bottom = np.concatenate((zeros, zeros), axis = 0)
                square = np.concatenate((top, bottom), axis = 1)
            elif channel == 1:
                top = np.concatenate((zeros,zeros), axis = 0)
                bottom = np.concatenate((ones, zeros), axis = 0)
                square = np.concatenate((top, bottom), axis = 1)  
            elif channel == 2:
                top = np.concatenate((zeros,ones), axis = 0)
                bottom = np.concatenate((zeros, zeros), axis = 0)
                square = np.concatenate((top, bottom), axis = 1)
            elif channel == 3:
                top = np.concatenate((zeros,zeros), axis = 0)
                bottom = np.concatenate((zeros, ones), axis = 0)
                square = np.concatenate((top, bottom), axis = 1)
            else:
                raise ValueError("Channel not in [0,1,2,3]")
            mask = tf.tile( square, (size[1]//(2*n), size[0]//(2*n)))
            return mask
        
        
        #self.mask = get_chess_mask(n, size)
        self.size = size
        self.masks = [ tf.reshape( get_chess_mask(n, size, i ), (size[0]*size[1]) ) for i in range(0,4)]
    
    
    
    def call(self, input,**kwargs): 
        image = input
        size_x = image.shape[2]
        size_y = image.shape[1]
        assert(size_x == size_y)
        assert(size_x == self.size[1])
        assert(size_y == self.size[0])


        #masks = [ tf.reshape( get_chess_mask(n, size_x, i ), (size_y*size_x) ) for i in range(0,4)]

        imgs = []
        image_res = tf.reshape(image, (-1,size_y*size_x))

        for m in self.masks:
            img = tf.boolean_mask(image_res, tf.cast(m, tf.bool), axis = 1)
            img = tf.reshape(img, (-1, size_y//2, size_x//2, 1))
            imgs.append(img)

        imgs = tf.concat(imgs, axis = 3)
        return imgs