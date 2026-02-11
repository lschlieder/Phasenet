import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PhaseplateNetwork.TFModules.PropagationLayer import PropagationLayer
from PhaseplateNetwork.TFModules.PropagationLayerVariableDistance import PropagationLayerVariableDistance
from PhaseplateNetwork.TFModules.OpticalLayers.WavePropagation import WavePropagation

epsilon = 1e-8


class cbn(tf.keras.layers.Layer):
    def __init__(self, n_kernels=10, act_func='linear'):
        super(cbn, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(n_kernels, (3, 3), activation=act_func, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(n_kernels, (3, 3), activation=act_func, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inp):
        x1 = self.bn1(self.conv1(inp))
        x1 = self.bn2(self.conv2(x1))
        return x1


class cc(tf.keras.layers.Layer):
    def __init__(self, n_kernels=10, act_func='linear'):
        super(cc, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(n_kernels, (3, 3), activation=act_func, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(n_kernels, (3, 3), activation=act_func, padding='same')

    def call(self, inp):
        return self.conv2(self.conv1(inp))

class PassThroughBlock(tf.keras.Model):
    def __init__(self, out = 4):
        super(PassThroughBlock,self).__init__()
        self.out = out
    def call(self,input):
        return input[:,:,:,0:self.out]


class AttentionBlock(tf.keras.Model):
    '''
    DotProductBlock for usage in ResNetGBNet
    '''
    def __init__(self, inp = 4, out = 4, out_activation = None):
        super(AttentionBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(inp, (1,1), padding = 'same')
        self.conv2 = tf.keras.layers.Conv2D(inp, (1,1), padding = 'same')
        self.conv3 = tf.keras.layers.Conv2D(inp, (1,1), padding = 'same')

        self.softmax = tf.keras.layers.Softmax(axis = [1,2])
        self.maxpool2d = tf.keras.layers.MaxPool2D((2,2), strides = 2)
        self.lastconv = tf.keras.layers.Conv2D(inp, (1,1), padding = 'same')

        self.output_conv = tf.keras.layers.Conv2D(out, (6,6),padding = 'same', activation  =out_activation)

        self.out_channels = out

        #self.conv_block = ConvBlock(out, out_activation)

        self.sigma = tf.Variable(0.0, trainable= True)

    def call(self,input):

        #print(input.shape)
        #feature_maps = self.conv_block(input)
        #input = self.conv_block(input)
        batch_size, h,w, c = input.shape
        #batch_size, h,w,c = feature_maps.shape
        dh = h//2
        dw = w//2
        #print(input.shape)

        f = self.conv1(input)
        #print(f.shape)
        #print(f.shape)
        #print(input.shape)
        #print([batch_size, h*w, c ])
        f = tf.reshape( f, [batch_size, h*w, c ])

        g = self.conv2(input)
        g = self.maxpool2d( g)
        g = tf.reshape(g, [batch_size, dh * dw, c])

        attn = tf.matmul(f, g, transpose_b = True)
        attn = self.softmax(attn)

        phi = self.conv3(input)
        phi = self.maxpool2d(phi)
        phi = tf.reshape( phi, [batch_size, dh * dw, c])

        attn_phi = tf.matmul(attn, phi)
        attn_phi = tf.reshape(attn_phi, [batch_size, h, w, c])
        attn_phi = self.lastconv(attn_phi)

        attn_phi = input + self.sigma*attn_phi
        res = self.output_conv(attn_phi)
        return res

class AttentionBlockConv(tf.keras.Model):
    '''
    DotProductBlock for usage in ResNetGBNet
    '''
    def __init__(self, inp = 4, out = 4, out_activation = None):
        super(AttentionBlockConv, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(out,  (1,1),padding = 'same')
        self.conv2 = tf.keras.layers.Conv2D(out, (1,1), padding = 'same')
        self.conv3 = tf.keras.layers.Conv2D(out, (1,1), padding = 'same')

        self.softmax = tf.keras.layers.Softmax(axis = [1,2])
        self.maxpool2d = tf.keras.layers.MaxPool2D((2,2), strides = 2)
        self.lastconv = tf.keras.layers.Conv2D(out, (1,1), padding = 'same')

        self.output_conv = tf.keras.layers.Conv2D(out, (6,6),padding = 'same', activation  =out_activation)

        self.out_channels = out

        self.conv_block = ConvBlock(out, out_activation)

        self.sigma = tf.Variable(0.0, trainable= True)

    def call(self,input):

        #print(input.shape)
        #feature_maps = self.conv_block(input)
        input = self.conv_block(input)
        batch_size, h,w, c = input.shape
        #batch_size, h,w,c = feature_maps.shape
        dh = h//2
        dw = w//2
        #print(input.shape)

        f = self.conv1(input)
        #print(f.shape)
        #print(f.shape)
        #print(input.shape)
        #print([batch_size, h*w, c ])
        f = tf.reshape( f, [batch_size, h*w, c ])

        g = self.conv2(input)
        g = self.maxpool2d( g)
        g = tf.reshape(g, [batch_size, dh * dw, c])

        attn = tf.matmul(f, g, transpose_b = True)
        attn = self.softmax(attn)

        phi = self.conv3(input)
        phi = self.maxpool2d(phi)
        phi = tf.reshape( phi, [batch_size, dh * dw, c])

        attn_phi = tf.matmul(attn, phi)
        attn_phi = tf.reshape(attn_phi, [batch_size, h, w, c])
        attn_phi = self.lastconv(attn_phi)

        attn_phi = input + self.sigma*attn_phi
        res = self.output_conv(attn_phi)
        return res




class SmallConvBlock(tf.keras.Model):

    def __init__(self, out = 4, out_activation = None):
        super(SmallConvBlock,self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(30,8,padding = 'same', activation ='relu')
        self.conv2 = tf.keras.layers.Conv2D(out, 10, padding = 'same', activation = out_activation)

    def call(self, input):
        out = self.conv2( self.bn2( self.conv1( self.bn1(input))))
        return out



class DeepCGHBlock(tf.keras.Model):
    def __init__(self, out = 4, out_activation = None, only_phase = False):
        super(DeepCGHBlock,self).__init__()

        self.only_phase=only_phase
        act_func = 'relu'
        n_kernels = [128,256,512]
        self.IF = 4
        self.cbn1 = cbn(n_kernels[0], act_func)
        self.maxp1 = tf.keras.layers.MaxPooling2D((2,2), padding = 'same')
        self.cbn2 = cbn(n_kernels[1], act_func)
        self.maxp2 = tf.keras.layers.MaxPooling2D((2,2), padding = 'same')
        self.cc1 = cc(n_kernels[2], act_func)
        self.ups1 = tf.keras.layers.UpSampling2D(2)
        self.conc1 = tf.keras.layers.Concatenate()
        self.cc2 = cc(n_kernels[1], act_func)
        self.ups2 = tf.keras.layers.UpSampling2D(2)
        self.conc2 = tf.keras.layers.Concatenate()
        self.cc3 = cc(n_kernels[0], act_func)
        self.cc4 = cc(n_kernels[1],act_func)
        self.conc3 = tf.keras.layers.Concatenate()
        self.conv1 = tf.keras.layers.Conv2D(self.IF**(2)*out, (3,3), activation = None, padding = 'same')
        self.conv2 = tf.keras.layers.Conv2D(self.IF**(2)*out, (3,3), activation ='tanh', padding = 'same')

    def call(self, inp):
        '''
        Calculates the output of the block
        .params: inputs: Input imagaes of shape [batch, x,y, channels]
        '''
        x1_1 = tf.nn.space_to_depth(input=inp, block_size=self.IF, data_format='NHWC')
        x1 = self.cbn1(x1_1)
        x2 = self.maxp1(x1)
        x2 = self.cbn2(x2)
        encoded = self.maxp2(x2)
        encoded = self.cc1(encoded)
        x3 = self.ups1(encoded)

        x3 = self.conc1([x3, x2])
        x3 = self.cc2(x3)

        x4 = self.ups2(x3)
        x4 = self.conc2([x4, x1])
        x4 = self.cc3(x4)

        x4 = self.cc4(x4)
        x4 = self.conc3([x4, x1_1])

        phi_0 = self.conv1(x4)
        phi_0 = tf.nn.depth_to_space(input=phi_0, block_size=self.IF, data_format='NHWC')

        amp_0 = np.pi* self.conv2(x4)
        amp_0 = tf.nn.depth_to_space(input=amp_0, block_size=self.IF, data_format='NHWC')

        if self.only_phase:
            out = phi_0
        else:
            out = tf.concat((amp_0, phi_0), axis = 3)
        return out

class ConvBlock(tf.keras.Model):
    '''
    Simple Convolution block for usage in ResNetGBNet
    :param out: output channels
    :param out_activation: keras activation string for the last layer
    '''
    def __init__(self, out = 4, out_activation = None, only_phase = False):


        super(ConvBlock,self).__init__()
        self.only_phase = only_phase

        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()
        self.batch_normalization3 = tf.keras.layers.BatchNormalization()
        self.batch_normalization4 = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.Conv2D(20, 8, padding = 'same', activation = 'relu')
        self.conv2 = tf.keras.layers.Conv2D(30, 5 , padding = 'same', activation = 'relu')
        self.conv3 = tf.keras.layers.Conv2D(40, 5, padding = 'same', activation = 'relu')
        self.conv4 = tf.keras.layers.Conv2D(50, 5, padding = 'same', activation = 'relu')
        self.conv_out_amp = tf.keras.layers.Conv2D(out, 3, padding = 'same',activation = 'relu')
        self.conv_out_phase = tf.keras.layers.Conv2D(out,3,padding = 'same',activation = out_activation)

    def call(self, inputs):
        '''
        Calculates the output of the block
        .params: inputs: Input imagaes of shape [batch, x,y, channels]
        '''
        #inp = tf.nn.space_to_depth(input=inputs, block_size=4, data_format='NHWC')
        inp = inputs
        out = self.conv4(self.batch_normalization4(self.conv3(self.batch_normalization3(self.conv2(self.batch_normalization2(self.conv1(self.batch_normalization1(inp))))))))
        #out = tf.nn.depth_to_space(input=out, block_size=4, data_format='NHWC')
        out_amp = self.conv_out_amp(out)
        #out_amp = tf.nn.depth_to_space(input = out_amp, block_size = 4, data_format='NHWC')
        out_phase = np.pi* self.conv_out_phase(out)
        #out_phase = tf.nn.depth_to_space(input = out_phase, block_size = 4, data_format='NHWC')
        #print(out_phase.shape)
        #print(inputs.shape)
        if self.only_phase:
            ret = out_phase
        else:
            ret = tf.concat([out_amp,out_phase],axis = 3)
        return ret

class BigConvBlock(tf.keras.Model):
    def __init__(self, out = 4, out_activation = None):
        super(BigConvBlock,self).__init__()
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
        out = self.conv_out(self.conv4(self.batch_normalization4(self.conv3(self.batch_normalization3(self.conv2(self.batch_normalization2(self.conv1(self.batch_normalization1(inputs)))))))))
        return out

class BiggestConvBlock(tf.keras.Model):
    def __init__(self, out = 4, out_activation = None):
        super(BiggestConvBlock,self).__init__()
        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()
        self.batch_normalization3 = tf.keras.layers.BatchNormalization()
        self.batch_normalization4 = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.Conv2D(100, 8, padding = 'same', activation = 'relu')
        self.conv2 = tf.keras.layers.Conv2D(100, 8 , padding = 'same', activation = 'relu')
        self.conv3 = tf.keras.layers.Conv2D(100, 8, padding = 'same', activation = 'relu')
        self.conv4 = tf.keras.layers.Conv2D(100, 8, padding = 'same', activation = 'relu')
        self.conv_out = tf.keras.layers.Conv2D(out, 20, padding = 'same',activation = out_activation)

    def call(self, inputs):
        out = self.conv_out(self.conv4(self.batch_normalization4(self.conv3(self.batch_normalization3(self.conv2(self.batch_normalization2(self.conv1(self.batch_normalization1(inputs)))))))))
        return out

class DenseBlock(tf.keras.Model):
    def __init__(self, out = 4, N = 60, out_activation = None):
        super(DenseBlock,self).__init__()
        #self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        #self.batch_normalization2 = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.Conv2D(1, 3, padding = 'same')
        self.conv2 = tf.keras.layers.Conv2D(1, 3, padding = 'same')
        self.conv3 = tf.keras.layers.Conv2D(1, 3, padding = 'same')

        self.conv_down1 = tf.keras.layers.Conv2D(30,8,2, padding = 'same', activation = 'relu')
        self.conv_down2 = tf.keras.layers.Conv2D(30,8,2, padding = 'same', activation = 'relu')
        self.conv_down3 = tf.keras.layers.Conv2D(30,8,2, padding = 'same', activation = 'relu')

        self.flatten = tf.keras.layers.Flatten()

        def get_downscaled_2(inp):

            for i in range(0,3):
                inp = np.ceil(inp/2)
            return int(inp)
        self.dense1 = tf.keras.layers.Dense(get_downscaled_2(N)**2 * 4, activation ='relu')
        #self.dense1 = tf.keras.layers.Dense(out*N*N, activation = 'relu')

        self.conv_up1 = tf.keras.layers.Conv2DTranspose(30,8,2, padding = 'same', activation  ='relu')
        self.conv_up2 = tf.keras.layers.Conv2DTranspose(30,8,2, padding = 'same', activation = 'relu')
        self.conv_up3 = tf.keras.layers.Conv2DTranspose(30,8,2, padding = 'same', activation ='relu')

        self.conv_output = tf.keras.layers.Conv2D(out,8, padding = 'same', activation = out_activation)

        #self.conv2 = tf.keras.layers.Conv2D(out, 20, padding = 'same')

    def call(self,inputs):
        #out = self.flatten( self.conv1( self.batch_normalization1(inputs)))
        #out = tf.reshape( self.dense1( self.batch_normalization2( out)), (-1, inputs.shape[1], inputs.shape[2], inputs.shape[3]))
        #out = self.conv2(self.dense1( self.batch_normalization2( self.conv1( self.batch_normalization1(inputs)))))

        #dense_out = tf.reshape( self.dense_inp(self.flatten(img)), (-1, self.N,self.N, 4))
        #print(dense_out.shape)
        d1 = self.conv_down1(inputs)
        d2 = self.conv_down2(d1)
        d3 = self.conv_down3(d2)
        #d4 = self.dblock4(d3)

        uf = self.flatten(d3)
        #print(d1.shape)
        #print(d2.shape)
        #print(d3.shape)
        #print(d4.shape)
        #print(uf.shape)
        up = tf.reshape(self.dense1(uf),(-1, d3.shape[1], d3.shape[2], 1))

        #up = self.conv5(d4)

        #u4 = tf.slice(self.ublock4(up),[0,0,0,0],[d3.shape[0], d3.shape[1], d3.shape[2], 1]) + self.conv4(d3)
        #print('test1')
        #print(d2.shape)
        #print(self.conv3(d2).shape)
        #print(self.conv_up3(up).shape)

        #print('test')
        u3 = tf.slice(self.conv_up3(up),[0,0,0,0],[d2.shape[0],d2.shape[1], d2.shape[2], 1]) + self.conv3(d2)
        u2 = tf.slice(self.conv_up2(u3),[0,0,0,0],[d1.shape[0], d1.shape[1], d1.shape[2], 1]) + self.conv2(d1)
        u1 = tf.slice(self.conv_up1(u2),[0,0,0,0],[inputs.shape[0], inputs.shape[1], inputs.shape[2],1]) + self.conv1(inputs)

        u = self.conv_output(u1)
        return u



class ProjectionBlockConv(tf.keras.Model):
    def __init__(self):
        super(ProjectionBlockConv,self).__init__()
        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.batch_normaliztion2 = tf.keras.layers.BatchNormalization()
        self.batch_normalization_out = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(10, 3, padding = 'same', activation = 'relu')
        self.conv2 = tf.keras.layers.Conv2D(5, 3, padding = 'same', activation = 'relu')
        #self.dense1 = tf.keras.layers.Dense(60*60)
        #self.flatten = tf.keras.layers.Flatten()

        self.conv_out = tf.keras.layers.Conv2D(1, 3, padding = 'same')


    def call(self,inputs):

        #out = self.dense1(self.flatten(self.batch_normalization_out(self.conv2(self.batch_normaliztion2(self.conv1(self.batch_normalization1(inputs)))))))
        #out = self.conv_out(tf.reshape(out, (-1, 60,60, 1))) + tf.expand_dims(tf.math.reduce_mean(inputs,axis = 3), axis = 3)
        out = self.conv_out(self.batch_normalization_out(self.conv2(self.batch_normaliztion2(self.conv1(self.batch_normalization1(inputs)))))) + tf.expand_dims(tf.math.reduce_mean(inputs,axis = 3), axis = 3)

        #u_new_c = tf.expand_dims(tf.math.reduce_mean(proj, axis=3), axis=3)


        return out

class ProjectionBlockDense(tf.keras.Model):
    def __init__(self):
        super(ProjectionBlockDense, self).__init__()
        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.batch_normaliztion2 = tf.keras.layers.BatchNormalization()
        self.batch_normalization_out = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(10, 3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(5, 3, padding='same', activation='relu')
        self.conv_down1 = tf.keras.layers.Conv2D(10, 3, 2, padding = 'same', activation = 'relu')
        self.conv_down2 = tf.keras.layers.Conv2D(10, 3, 2, padding = 'same', activation = 'relu')

        self.dense1 = tf.keras.layers.Dense(60 * 60, activation = 'relu')
        self.flatten = tf.keras.layers.Flatten()

        self.batch_normalization_out1 = tf.keras.layers.BatchNormalization()
        self.batch_normalization_out2 = tf.keras.layers.BatchNormalization()
        self.conv_up1 = tf.keras.layers.Conv2DTranspose(10,3, 2, padding = 'same', activation = 'relu')
        self.conv_up2 = tf.keras.layers.Conv2DTranspose(10,3,2, padding = 'same', activation = 'relu')

        self.conv_out = tf.keras.layers.Conv2D(1, 3, padding='same')

    def call(self, inputs):

        out = self.dense1(self.flatten(self.conv_down2( self.batch_normalization2( self.conv_down1( self.batch_normalization1( inputs))))))

        out = self.conv_out( self.conv_up2( self.batch_normalization_out2( self.conv_up1(self.batch_normalization_out1(out)))))

        out = self.dense1(self.flatten(self.batch_normalization_out(self.conv2(self.batch_normaliztion2(self.conv1(self.batch_normalization1(inputs)))))))
        out = self.conv_out(tf.reshape(out, (-1, 60,60, 1))) + tf.expand_dims(tf.math.reduce_mean(inputs,axis = 3), axis = 3)
        #out = self.conv_out(self.batch_normalization_out(
        #    self.conv2(self.batch_normaliztion2(self.conv1(self.batch_normalization1(inputs)))))) + tf.expand_dims(
            #tf.math.reduce_mean(inputs, axis=3), axis=3)

        # u_new_c = tf.expand_dims(tf.math.reduce_mean(proj, axis=3), axis=3)

        return out


class ResNetForwardBackwardSplittingModel(tf.keras.Model):


    def __init__(self, z, L, N, transducer_radius=25, padding=50, depth=20, f0=2.00e6, cM=1484, use_IASA=False,

                 layer_type='conv', activation_type = 'tanh', random_distance = False, use_residual_connection = False):
        '''
        Create The Residual Hologram Network model
        :param z : list of hologram planes to be created
        :param L : Side length of the Hologram in [mm]
        :param N : Pixel on the Hologram and output plane
        '''
        super(ResNetForwardBackwardSplittingModel, self).__init__()

        self.L = L
        self.N = N
        self.padding = padding
        self.z = z
        self.use_IASA = use_IASA
        self.layer_type = layer_type
        self.transducer_radius = transducer_radius
        self.f0 = f0
        self.cM = cM
        dx = L / N
        X, Y = np.mgrid[-L / 2:L / 2:dx, -L / 2:L / 2:dx]

        print('Pixel_size:{}'.format(dx))
        amplitude = np.zeros_like(X)
        amplitude[X ** 2 + Y ** 2 < transducer_radius ** 2] = 1.0
        self.amplitude = np.reshape(amplitude, (amplitude.shape[0], amplitude.shape[1], 1))

        self.random_distance = random_distance

        self.use_residual_connection = use_residual_connection

        self.conv_blocks_forward = []
        self.conv_blocks_backward = []
        for i in range(0,depth):
            print(layer_type)

            if layer_type == 'conv':
                self.conv_blocks_forward.append( ConvBlock(len(z), out_activation = activation_type))
                self.conv_blocks_backward.append(ConvBlock(1, out_activation=activation_type, only_phase = True))
            elif layer_type == 'small_conv':
                self.conv_blocks_forward.append( SmallConvBlock(len(z)*2))
                self.conv_blocks_backward.append( SmallConvBlock(1, out_activation=activation_type))
            elif layer_type =='bigconv':
                self.conv_blocks_forward.append(BigConvBlock(len(z)*2))
                self.conv_blocks_backward.append(BigConvBlock(1, out_activation = activation_type))
            elif layer_type =='biggest_conv':
                self.conv_blocks_forward.append(BiggestConvBlock(len(z)*2))
                self.conv_blocks_backward.append(BiggestConvBlock(1, out_activation = activation_type))
            elif layer_type == 'dense':
                self.conv_blocks_forward.append( DenseBlock(len(z)*2, N))
                self.conv_blocks_backward.append(DenseBlock(1, N, out_activation=activation_type))
            elif layer_type == 'attention':
                self.conv_blocks_forward.append( AttentionBlock( len(z)*3, len(z)*2))
                self.conv_blocks_backward.append( AttentionBlock( len(z)*2+1, 1, out_activation = activation_type ))
            elif layer_type == 'attention_conv':
                self.conv_blocks_forward.append( AttentionBlockConv( len(z)*3, len(z)*2))
                self.conv_blocks_backward.append( AttentionBlockConv( len(z)*2+1, 1, out_activation = activation_type ))

            elif layer_type == 'deepcgh':
                self.conv_blocks_forward.append( DeepCGHBlock(len(z)*2))
                #self.conv_blocks_backward.append( PassThroughBlock())
                self.conv_blocks_backward.append(DeepCGHBlock(1, only_phase= True))

            else:
                print('Layer type {} not found'.format(layer_type))
                raise RuntimeError('Layer type not found')




        self.z = tf.cast(z, dtype=tf.float32)
        self.network_depth = depth
        self.propagation_layers = []
        for z_i in z:
            if random_distance == False:
                #self.propagation_layers.append(PropagationLayer(z_i, N, L, padding = padding, f0 = f0, cM = cM))

                self.propagation_layers.append(WavePropagation(z_i, N, L, padding = padding, f0 = f0, cM = cM))
                #(self, z, N, L, scale
                 #= 2, padding = None, f0=2.00e6, cM=1484, channels_last=True, use_FT=True, ** kwargs):
            else:
                self.propagation_layers.append(PropagationLayerVariableDistance(z_i, N, L, padding, f0, cM))

    def get_network_hyperparameters(self):
        '''
        Get the network hyperparameters
        :return: This returns all parameters that are needed to reproduce the network in the real world. Training options are not returned
        '''
        str = 'Network hyperparameters and physical dimensions:\n'
        str = str + '\n'
        # str = str + 'Propagation area size: {} mm\n'.format(self.L)
        str = str + 'Plate size: {} mm\n'.format(self.L)
        # str = str + 'Complete Pixel count (pixel in propagation area): {}\n'.format(self.N)
        #str = str + 'Hologram pixel count (pixel on plate): {}\n'.format(self.Trainable_Pixel)
        str = str + 'Trainable pixel on plate (are upscaled to match pixel on plate): {}\n'.format(self.N)
        #str = str + 'Trainable pixel size: {} mm\n'.format(self.dp)
        #str = str + 'Image pixel size: {} mm\n'.format(self.dp_im)
        str = str + 'Transducer radius: {} mm\n'.format(self.transducer_radius)
        str = str + 'Transducer frequency: {} Hz\n'.format(self.f0)
        str = str + 'Wavespeed in medium: {} mm/s\n'.format(self.cM)
        #str = str + 'Distance between transducer and first plate: {}\n'.format(self.first_layer_distance)
        #str = str + 'Distance between last plate and detector: {}\n'.format(self.last_layer_distance)
        str = str + 'Plate Distance:\n'
        i = 1
        for dist in self.z:
            str = str + 'layer{}: {}\n'.format(i, dist)
            i = i + 1
        i = 1

        str = str + 'Network Depth: {}\n'.format(self.network_depth)
        str = str + 'Layer Type:{}\n'.format(self.layer_type)

        #str = str + 'Fully connected distance: {}\n \n'.format(self.fully_connected_distance)
        return str

    @tf.function
    def projection_to_image_constraint(self, u, img):
        #u = self.amplitude * tf.exp(1j*tf.cast(tf.math.angle(u),dtype=tf.complex64))

        projected_u_arr = []
        i = 0
        for layer in self.propagation_layers:
            #print(u.shape)
            prop_u = layer(u)
            #print(prop_u.shape)
            #propagated_u.append(prop_u)
            img_i = tf.expand_dims(img[:,:,:,i], axis = 3)
            projected_u = tf.cast(img_i+epsilon,dtype = tf.complex64) * tf.math.exp(1j*tf.cast(tf.math.angle(prop_u),dtype= tf.complex64))
            #print('proj u shape')
            #print(projected_u.shape)
            reprop_u = layer.inverse_call(projected_u)
            #print(reprop_u.shape)
            projected_u_arr.append(reprop_u)
            i = i+1

        #print(len(projected_u_arr))
        return tf.concat(projected_u_arr, axis = 3)

    @tf.function
    def forward_propagate(self, u,z= 0):
        projected_u_arr = []
        i = 0
        for layer in self.propagation_layers:
            if self.random_distance:
                prop_u = layer(u,z)
            else:
                prop_u = layer(u)
            projected_u_arr.append(prop_u)
        return tf.concat(projected_u_arr, axis = 3)

    @tf.function
    def backward_propagate(self,u,z = 0 ):
        projected_u_arr = []
        i = 0
        for layer in self.propagation_layers:
            if self.random_distance:
                prop_u = layer.inverse_call(u[:,:,:,i:i+1],z)
            else:
                prop_u = layer.inverse_call(u[:,:,:,i:i+1])

            projected_u_arr.append(prop_u)
            i = i +1
        return tf.concat(projected_u_arr, axis = 3)

    @tf.function
    def project_to_transducer_constraint(self, u):
        return tf.cast(self.amplitude+epsilon,dtype=tf.complex64) * tf.math.exp(1j*tf.cast(tf.math.angle(u), dtype = tf.complex64))

    # @tf.function
    def call(self, inputs, **kwargs):
        img = inputs
        self.amplitude_tiled = tf.cast(tf.tile(tf.reshape(self.amplitude, (1,self.amplitude.shape[0], self.amplitude.shape[1], 1)), (inputs.shape[0], 1,1,1)),dtype = tf.float32)
        u = tf.constant(np.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2], 1)))
        u = (self.amplitude + epsilon) * tf.exp(1j * tf.cast(u, dtype=tf.complex64))
        u_array = []

        u_res = u
        z = tf.random.uniform([u.shape[3]], 20.0, 50.0)
        for i in range(0, self.network_depth):
            #proj = self.projection_to_image_constraint(u, img)
            if self.random_distance:
                proj = complex_to_channel(self.forward_propagate(u,z))
                proj = tf.concat((proj, img), axis = 3)
                image_plane = channel_to_complex(self.conv_blocks_forward[i].call(proj))
                back_proj = complex_to_channel(self.backward_propagate(image_plane,z))
                back_proj = tf.concat((back_proj, tf.math.angle(u)), axis = 3)
                hologram_plane = np.pi*( self.conv_blocks_backward[i].call(back_proj))
                if self.use_residual_connection:
                    u =  (self.amplitude + epsilon) * tf.exp(1j * tf.cast(hologram_plane + tf.math.angle(u), dtype=tf.complex64))
                else:
                    u = (self.amplitude + epsilon) * tf.exp(1j * tf.cast(hologram_plane, dtype=tf.complex64))
            else:

                proj = complex_to_channel(self.forward_propagate(u))
                proj_img = tf.concat((proj, img), axis = 3)
                #if self.use_residual_connection:
                    #print(i)
                    #print(proj.shape)
                    #print(self.conv_blocks_forward[i].call(proj_img).shape)
                 #   image_plane = channel_to_complex( proj + self.conv_blocks_forward[i].call(proj_img))
                #else:
                image_plane = channel_to_complex(self.conv_blocks_forward[i].call(proj_img))
                #print(proj.dtype)
                #image_plane = channel_to_complex(self.conv_blocks_forward[i].call(proj_img))

                #holo_plane = self.backward_propagate(image_plane)

                back_proj = complex_to_channel(self.backward_propagate(image_plane))

                #back_proj_u = tf.concat((back_proj, tf.math.angle(u)), axis = 3)

                back_proj_u = tf.concat((back_proj, self.amplitude_tiled), axis = 3)
                #if self.use_residual_connection:
                #print('plane output: {}'.format(self.conv_blocks_backward[i].call(back_proj_u).shape))
                #    hologram_plane = ( back_proj +self.conv_blocks_backward[i].call(back_proj_u))
                #else:
                #hologram_plane = self.conv_blocks_backward[i].call(back_proj_u)[:,:,:,0:1]
                hologram_plane = self.conv_blocks_backward[i].call(back_proj_u)

                #hologram_plane = tf.expand_dims(tf.math.reduce_mean(tf.math.angle(back_proj), axis = 3),axis = 3)


                #hologram_plane = np.pi* channel_to_complex(hologram_plane)
                if self.use_residual_connection:
                    #print(hologram_plane.shape)
                    u =  (self.amplitude + epsilon) * tf.exp(1j * (tf.cast(hologram_plane + tf.math.angle(u), dtype=tf.complex64)))
                    #print(u.shape)
                else:
                    u = (self.amplitude + epsilon) * tf.exp(1j * tf.cast(hologram_plane, dtype=tf.complex64))
            #print(back_proj.shape)
            #print(hologram_plane.shape)
            #u = self.project_to_transducer_constraint(out)
            #(self.amplitude + epsilon) * tf.exp(1j * tf.cast(out, dtype=tf.complex64))

            #u = (self.amplitude + epsilon) * tf.exp(1j* tf.cast(tf.math.angle(u), dtype = tf.complex64))

            #u = (self.amplitude + epsilon)*u
            u_array.append(tf.math.angle(u))
            #plt.imshow(np.abs(u[0,:,:,0]))
            #plt.show()


            #proj = complex_to_channel(proj)
            #print(proj.shape)
            #proj = tf.concat((tf.math.angle(proj),img, tf.math.angle(u)), axis = 3)
            #out = self.conv_blocks[i].call(proj)
            #print(out.shape)
            #out = channel_to_complex(out)
            #print(out.shape)
            #out = (self.amplitude + epsilon) * tf.exp(1j * tf.cast(out, dtype=tf.complex64))
            #u = self.project_to_transducer_constraint(out)

            #u_array.append(tf.math.angle(u))

        #result = tf.stack(u_array)

        u = (self.amplitude + epsilon) * tf.exp(1j * tf.cast(tf.math.angle(u), dtype=tf.complex64))

        prop_u = []
        for layer in self.propagation_layers:
            if self.random_distance:
                prop_u.append(layer(u,z))
            else:
                prop_u.append(layer(u))
        res = tf.concat(prop_u, axis=3)

        self.u_array = [u]
        return res, [u]

    def get_propagation(self):
        return self.u_array


# takes a tensor of rank 4 ( batch, imgx, imgy, n) and returns a tensor of dimensions (batch, imgx, imgy, 2*n ) with the imaginary and real part as channels
@tf.function
def complex_to_channel(input, channel_dim=3):
    real = tf.math.real(input)
    imag = tf.math.imag(input)
    return tf.concat([real, imag], channel_dim)


@tf.function
def complex_to_channel_radial(input, channel_dim=3):
    abs = tf.math.abs(input)
    angle = tf.math.angle(input)
    return tf.concat([abs, angle], channel_dim)


@tf.function
def channel_to_complex(input, channel_dim=3):
    slice_arr_real1 = [0, 0, 0, 0]
    slice_arr_real2 = [input.shape[0], input.shape[1], input.shape[2], input.shape[3]]
    slice_arr_real2[channel_dim] = input.shape[channel_dim] // 2
    slice_arr_imag1 = [0, 0, 0, 0]
    slice_arr_imag1[channel_dim] = input.shape[channel_dim] // 2
    slice_arr_imag2 = [input.shape[0], input.shape[1], input.shape[2], input.shape[3]]
    slice_arr_imag2[channel_dim] = input.shape[channel_dim] // 2
    # print(slice_arr_imag1)
    # print(slice_arr_imag2)
    real = tf.slice(input, slice_arr_real1, slice_arr_real2)
    imag = tf.slice(input, slice_arr_imag1, slice_arr_imag2)
    ret = tf.complex(real, imag)
    return ret


# @tf.function
def channel_to_complex_radial(input, channel_dim=3):
    slice_arr_abs1 = [0, 0, 0, 0]
    slice_arr_abs2 = [input.shape[0], input.shape[1], input.shape[2], input.shape[3]]
    slice_arr_abs2[channel_dim] = input.shape[channel_dim] // 2
    slice_arr_angle1 = [0, 0, 0, 0]
    slice_arr_angle1[channel_dim] = input.shape[channel_dim] // 2
    slice_arr_angle2 = [input.shape[0], input.shape[1], input.shape[2], input.shape[3]]
    slice_arr_angle2[channel_dim] = input.shape[channel_dim] // 2
    abs_img = tf.slice(input, slice_arr_abs1, slice_arr_abs2)
    angle = tf.slice(input, slice_arr_angle1, slice_arr_angle2)
    ret = tf.complex(abs_img + epsilon, 0.0) * tf.math.exp(tf.complex(0.0, angle))

    return ret


@tf.function
def compute_gradients_and_perform_update_step(model, optimizer, u, holo, z, images):
    with tf.GradientTape() as g:
        result = model.call((u, images))
        result = tf.transpose(result, (0, 1, 4, 2, 3))
        propagated_images = holo.PropagateToPlaneTF(result, z)
        images = tf.reshape(images, (1, images.shape[0], images.shape[1], images.shape[2], images.shape[3]))
        images = tf.tile(images, (model.network_depth, 1, 1, 1, 1))
        images = tf.transpose(images, (0, 1, 4, 2, 3))
        # print(images)
        # print(propagated_images)
        loss = tf.reduce_mean(tf.math.abs(propagated_images) - tf.math.abs(images))
    gradients = g.gradient(loss, model.weights)
    optimizer.apply_gradients(zip(gradients, model.weights))
    # print(pred_y.shape)
    return loss, result, holo.PropagateToPlaneTF(result, z), im