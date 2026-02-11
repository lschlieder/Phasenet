import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PhaseplateNetwork.TFModules.PropagationLayer import PropagationLayer

epsilon = 1e-10


class ResModel(tf.keras.Model):

    def __init__(self, z, L, N, transducer_radius = 25, padding = 50, depth = 20, f0 = 2.00e6, cM = 1484, use_IASA = False, layer_type = 'big'):
        '''
        Create The Residual Hologram Network model
        :param z :
        '''
        super(ResModel, self).__init__()
        #self.holo = Hologram()
        #self.single_layer = create_single_layer()
        self.L = L
        self.N = N
        self.padding = padding
        self.z = z
        self.use_IASA = use_IASA
        self.layer_type = layer_type
        dx = L/N
        X,Y = np.mgrid[-L/2:L/2:dx, -L/2:L/2:dx]
        amplitude = np.zeros_like(X)
        amplitude[X**2 + Y**2 < transducer_radius**2] = 1.0
        self.amplitude = np.reshape(amplitude, (amplitude.shape[0],amplitude.shape[1],1))
        #plt.imshow(amplitude)
        #plt.show()
        value = np.zeros((1, 1, 8, 2))
        value = value + epsilon

        #value[0,0,4:8,1] = 0.25
        value[0,0,4:8,0] = 0.25

        #value[0,0,3,1] = 0.5
        init = tf.initializers.constant(value)
        self.conv1 = tf.keras.layers.Conv2D(20, (5, 5), padding='same', activation='relu',use_bias = False)
        self.conv2 = tf.keras.layers.Conv2D(30, (5, 5), padding='same', activation='relu', use_bias = False)
        self.conv4 = tf.keras.layers.Conv2D(40, (5,5), padding = 'same', activation = 'relu', use_bias = False)
        self.conv5 = tf.keras.layers.Conv2D(40, (5,5), padding = 'same', activation = 'relu', use_bias = False)
        self.conv3 = tf.keras.layers.Conv2D(2, (5,5), padding = 'same', use_bias = False)
        self.conv3_relu = tf.keras.layers.Conv2D(2, (5,5), padding = 'same', use_bias = False, activation = 'relu')
        self.dense1 = tf.keras.layers.Dense(N*N*2, activation = 'relu')
        self.flatten = tf.keras.layers.Flatten()
        self.conv_average = tf.keras.layers.Conv2D(2, (1,1), padding = 'same', use_bias = False)

        #weights = tf.constant(1.0, shape = (2))
        #self.weights = tf.Variable(weights)


        #self.flatten = tf.keras.layers.Flatten()
        #self.dense = tf.keras.layers.Dense(60*60*2, activation='tanh')

        #value = np.zeros((5,5, images*2, 2))
        #value[2,2] = 1.0
        #init = tf.initializers.constant(value)
        #self.convHIO = tf.keras.layers.Conv2D(2, (5,5),padding = 'same', kernel_initializer = init)
        #self.beta = tf.Variable(0.9,True, dtype = tf.float32)
        
        #self.support_region = tf.cast(support_region, dtype = tf.float32)
        #zeros = tf.constant(0.0, shape = self.support_region.shape)
        #self.support_region = tf.complex(self.support_region, zeros)
        self.z = tf.cast(z, dtype=tf.float32)
        self.network_depth = depth
        self.propagation_layers = []
        for z_i in z:
            self.propagation_layers.append(PropagationLayer(z_i, N, L , padding, f0, cM))


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
    def project_to_transducer_constraint(self, u):
        return tf.cast(self.amplitude+epsilon,dtype=tf.complex64) * tf.math.exp(1j*tf.cast(tf.math.angle(u), dtype = tf.complex64))

    @tf.function
    def single_layer_pass_IASA(self, u, img):
        proj = self.projection_to_image_constraint(u, img)
        #inp = tf.concat([u, proj], 3)
        #print(proj.shape)
        ############################### with machine learning
        #inp_c = complex_to_channel(proj)
        #
        #u_new = self.conv3(inp_c)
        #u_new_c = channel_to_complex(u_new)
        #######
        #without machine learning
        u_new_c = tf.expand_dims(tf.math.reduce_mean(proj, axis = 3), axis = 3)
        #print(u_new_c.shape)

        u = self.project_to_transducer_constraint(u_new_c)
        return u


    def single_layer_pass_dense(self, u, img):
        proj = self.projection_to_image_constraint(u, img)
        inp_c = complex_to_channel(proj)
        dense_out = tf.reshape(self.dense1(self.flatten(inp_c)), (-1, self.N, self.N, 2))
        u_new = self.conv3(dense_out)
        u_new_c =channel_to_complex_radial(u_new)
        res = u_new_c
        u = self.project_to_transducer_constraint(res)
        return u

    def single_layer_pass_average(self, u , img):
        proj = self.projection_to_image_constraint(u,img)
        inp_c = complex_to_channel_radial(proj)
        u_new = self.conv_average(inp_c)
        u_new_c = channel_to_complex_radial(u_new)
        #plt.imshow(tf.math.angle(u_new_c[0,:,:,0]))
        #plt.figure()
        #plt.imshow(tf.math.angle(tf.math.reduce_mean(proj, axis = 3)[0]))
        #plt.show()
        #res = u_new_c
        res = tf.complex(tf.math.abs(u) + tf.math.abs(u_new_c) + epsilon, 0.0) * tf.math.exp(
            tf.complex(0.0, tf.math.angle(u) + tf.math.angle(u_new_c)))
        u = self.project_to_transducer_constraint(res)
        return u

    def single_layer_pass_small(self, u, img):
        proj = self.projection_to_image_constraint(u, img)
        inp_c = complex_to_channel_radial(proj)
        u_new = self.conv3(inp_c)
        u_new_c =channel_to_complex_radial(u_new)
        res = u_new_c
        u = self.project_to_transducer_constraint(res)
        return u

    def single_layer_pass_small_relu(self, u, img):
        proj = self.projection_to_image_constraint(u, img)
        inp_c = complex_to_channel_radial(proj)
        u_new = self.conv3_relu(inp_c)
        u_new_c =channel_to_complex_radial(u_new)
        res = u_new_c
        u = self.project_to_transducer_constraint(res)
        return

    def single_layer_pass_small_res(self, u, img):
        proj = self.projection_to_image_constraint(u, img)
        inp_c = complex_to_channel_radial(proj)
        u_new = self.conv3(inp_c)
        u_new_c =channel_to_complex_radial(u_new)
        #res = u_new_c
        res = tf.complex(tf.math.abs(u) + tf.math.abs(u_new_c) + epsilon, 0.0) * tf.math.exp(
            tf.complex(0.0, tf.math.angle(u) + tf.math.angle(u_new_c)))
        u = self.project_to_transducer_constraint(res)
        return u
    #@tf.function

    def single_layer_pass_big(self, u, img):
        #plt.imshow(tf.math.abs(u[0,:,:,0]))
        #plt.figure()
        #plt.imshow(tf.math.angle(u[0,:,:,0]))
        #plt.show()
        proj = self.projection_to_image_constraint(u, img)
        #print(np.max(tf.math.angle(proj)))
        #print(np.min(tf.math.angle(proj)))
        #inp = tf.concat([u, proj], 3)
        #print(proj.shape)
        ############################### with machine learning
        #inp_c = tf.math.angle(proj)
        inp_c = complex_to_channel_radial(proj)
        #
        #u_new = self.conv3(self.conv2(self.conv1(inp_c)))
        #u_new = self.conv3(self.conv2(self.conv1(inp_c)))*2*np.pi
        u_new = self.conv3((self.conv2(self.conv1(inp_c))))
        #u_new = self.conv3(inp_c)*2*np.pi
        #temp = self.dense(self.flatten(inp_c))*2*np.pi

        #u_new = tf.reshape(self.dense(self.flatten(inp_c))*2*np.pi, (inp_c.shape[0], inp_c.shape[1], inp_c.shape[2], proj.shape[3]))

        #inp_c_slice_abs = inp_c[:,:,:,0:inp_c.shape[3]//2]
        #inp_c_angle = inp_c[:,:,:,inp_c.shape[3]//2:inp_c.shape[3]]
        #print(inp_c_angle.shape)
        #u_new = self.conv_average(inp_c)
        #print(u_new.shape)
        #inp_c_stacked = tf.concat((inp_c_slice_abs[:,:,:,0:1],u_new), axis = 3)
        #print(inp_c_stacked.shape)
        #print(inp_c_stacked.shape)
        #u_new = self.conv3(inp_c)*2*np.pi
        u_new_c = channel_to_complex_radial(u_new)
        #print('u_new_c shape')
        #print(u_new_c.shape)
        #u_new_c = tf.complex(0.0,0.0) * tf.math.exp(tf.complex(0.0,u_new))
        #######
        #without machine learning
        #u_new_c = tf.expand_dims(tf.math.reduce_mean(proj, axis = 3), axis = 3)
        #print(u_new_c.shape)
        #res = tf.complex(tf.math.abs(u) +tf.math.abs(u_new_c)+epsilon,0.0) * tf.math.exp( tf.complex( 0.0, tf.math.angle(u) + tf.math.angle(u_new_c)))
        res = u_new_c
        #plt.imshow(tf.math.angle(res)[0,:,:,0])
        #plt.show()
        #u = self.project_to_transducer_constraint(u_new_c + u)

        u = self.project_to_transducer_constraint(res)
        '''
        plt.figure()
        plt.imshow(tf.math.angle(proj[0,:,:,0]))
        plt.figure()
        plt.imshow(tf.math.angle(proj[0,:,:,1]))
        plt.figure()
        plt.imshow(tf.math.angle(res[0,:,:,0]))
        plt.figure()
        plt.imshow(tf.math.reduce_mean( tf.math.angle(proj),axis = 3)[0])
        plt.show()
        '''
        #print(np.max(tf.math.angle(u)))
        #print(np.min(tf.math.angle(u)))
        return u

    def single_layer_pass_big_res(self, u, img):
        proj = self.projection_to_image_constraint(u, img)
        inp_c = complex_to_channel_radial(proj)
        u_new = self.conv3((self.conv2(self.conv1(inp_c))))
        u_new_c = channel_to_complex_radial(u_new)
        #res = tf.complex(tf.math.abs(u) +tf.math.abs(u_new_c)+epsilon,0.0) * tf.math.exp( tf.complex( 0.0, tf.math.angle(u) + tf.math.angle(u_new_c)))
        res = u + u_new_c
        #res = u_new_c
        u = self.project_to_transducer_constraint(res)
        return u


    #@tf.function
    #def




    @tf.function
    def single_layer_pass_HIONet(self, u, img):
        #proj = self.holo.get_projection_to_M(u,img,self.z, True)
        proj = self.project_to_transducer_constraint(self.projection_to_image_constraint(u, img))
        #print(proj.shape)
        #print(proj.dtype)
        proj_c = complex_to_channel(proj)
        #print(proj_c.shape)
        #print(proj.dtype)
        conv_proj_c = self.convHIO(proj_c)
        conv_proj = channel_to_complex(conv_proj_c)
        #support_reg = tf.reshape(self.support_region, (1, conv_proj.shape[1], conv_proj.shape[2], 1))
        #support_reg = tf.tile(support_reg, (conv_proj.shape[0], 1, 1, 1))
        #other_phase = (tf.cast(self.beta,tf.complex64)* conv_proj + u)* tf.cast(tf.math.abs(support_reg - 1.0), dtype = tf.complex64)
        #support_phase = conv_proj* support_reg
        u_new = other_phase + support_phase
        return u_new

    #def build(self,input_shape):
        #print(input_shape)
        #self.amplitude = tf.cast( tf.reshape( self.amplitude, (self.N,self.N,input_shape[3])),tf.float32)

        #amplitude = tf.complex(amplitude, tf.zeros(amplitude.shape, dtype=tf.float32))
        #self.amplitude = amplitude
        #self.amplitude = tf.tile( self.amplitude,(1,1,input_shape[3]))
    #@tf.function
    def call(self, inputs, **kwargs):
        #print(inputs.shape)
        img = inputs

        u = tf.constant(np.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2], 1)))
        u = (self.amplitude + epsilon) * tf.exp(1j*tf.cast(tf.math.angle(u),dtype=tf.complex64))
        u_array = []

        u_res = u
        for i in range(0, self.network_depth):
            #u = u+ self.single_layer_pass(u, img)
            if self.layer_type == 'small':
                u = self.single_layer_pass_small(u,img)
            elif self.layer_type == 'small_relu':
                u = self.single_layer_pass_small_relu(u,img)
            elif self.layer_type =='small_res':
                if i % 4 == 0:

                    u_new = self.single_layer_pass_small_res(u,img)
                    u = tf.complex(tf.math.abs(u_res) + tf.math.abs(u_new) + epsilon, 0.0) * tf.math.exp(
                        tf.complex(0.0, tf.math.angle(u_res) + tf.math.angle(u_new)))
                    u_res = u
                    #u = self.single_layer_pass_small_res(u,img)
                else:
                    u = self.single_layer_pass_small_res(u, img)
            elif self.layer_type == 'big_res':
                u = self.single_layer_pass_big_res(u,img)
            elif self.layer_type == 'big':
                u = self.single_layer_pass_big(u, img)
            elif self.layer_type == 'average':
                u = self.single_layer_pass_average(u,img)
            elif self.layer_type == 'dense':
                u = self.single_layer_pass_dense(u, img)
            elif self.layer_type == 'IASA':
                u = self.single_layer_pass_IASA(u, img)
            #plt.imshow(tf.math.angle(u[0,:,:,0]))
            #plt.show()
            #print(u.shape)
            u_array.append(tf.math.angle(u))
        result = tf.stack(u_array)
        #print(np.max(tf.math.angle(u)))
        #print(np.min(tf.math.angle(u)))
        prop_u = []
        for layer in self.propagation_layers:
            #print(u.shape)
            prop_u.append( layer(u))
        res = tf.concat(prop_u,axis = 3)
        #print(tf.math.reduce_sum(tf.cast(tf.math.is_nan(tf.abs(res)),dtype = tf.int8)).numpy())
        #print('res shape')
        #print(res.shape)
        #print(result.shape)
        self.u_array = [u]
        return res, [u]

    def get_propagation(self):
        return self.u_array


#takes a tensor of rank 4 ( batch, imgx, imgy, n) and returns a tensor of dimensions (batch, imgx, imgy, 2*n ) with the imaginary and real part as channels
@tf.function
def complex_to_channel( input, channel_dim = 3):
    real = tf.math.real(input)
    imag = tf.math.imag(input)
    return tf.concat([real,imag], channel_dim)

@tf.function
def complex_to_channel_radial(input, channel_dim = 3):
    abs = tf.math.abs(input)
    angle = tf.math.angle(input)
    return tf.concat([abs,angle], channel_dim)

@tf.function
def channel_to_complex(input, channel_dim = 3):
    slice_arr_real1 = [0,0,0,0]
    slice_arr_real2 = [input.shape[0], input.shape[1], input.shape[2], input.shape[3]]
    slice_arr_real2[channel_dim] = input.shape[channel_dim]//2
    slice_arr_imag1 = [0,0,0,0]
    slice_arr_imag1[channel_dim] = input.shape[channel_dim]//2
    slice_arr_imag2 = [input.shape[0],input.shape[1], input.shape[2],input.shape[3]]
    slice_arr_imag2[channel_dim] = input.shape[channel_dim]//2
    #print(slice_arr_imag1)
    #print(slice_arr_imag2)
    real = tf.slice(input, slice_arr_real1, slice_arr_real2)
    imag = tf.slice(input, slice_arr_imag1, slice_arr_imag2)
    ret = tf.complex( real,imag)
    return ret

#@tf.function
def channel_to_complex_radial(input, channel_dim = 3):
    slice_arr_abs1 = [0, 0, 0, 0]
    slice_arr_abs2 = [input.shape[0], input.shape[1], input.shape[2], input.shape[3]]
    slice_arr_abs2[channel_dim] = input.shape[channel_dim] // 2
    slice_arr_angle1 = [0, 0, 0, 0]
    slice_arr_angle1[channel_dim] = input.shape[channel_dim] // 2
    slice_arr_angle2 = [input.shape[0], input.shape[1], input.shape[2], input.shape[3]]
    slice_arr_angle2[channel_dim] = input.shape[channel_dim] // 2
    # print(slice_arr_imag1)
    # print(slice_arr_imag2)
    abs_img = tf.slice(input, slice_arr_abs1, slice_arr_abs2)
    angle = tf.slice(input, slice_arr_angle1, slice_arr_angle2)
    #plt.imshow(angle[0,:,:,0])
    #plt.show()
    #print(angle.dtype)
    #ret = tf.cast(abs_img, dtype = tf.complex64)*tf.math.exp(1j*tf.cast(angle, dtype = tf.complex64))

    ret = tf.complex(abs_img + epsilon,0.0) * tf.math.exp(tf.complex(0.0,angle))
    #print(ret.dtype)
    #plt.figure()
    #plt.imshow(tf.math.angle(ret[0,:,:,0]))
    #plt.show()
    #print(tf.math.reduce_sum(tf.cast(tf.math.is_nan(tf.math.abs(ret)),dtype = tf.int8)).numpy())
    #print(tf.math.reduce_sum(tf.cast(tf.math.is_nan(tf.math.angle(ret)),dtype = tf.int8)).numpy())
    return ret


@tf.function
def compute_gradients_and_perform_update_step(model, optimizer,u,holo,z,images ):
    with tf.GradientTape() as g:
        result = model.call((u,images))
        result = tf.transpose( result, ( 0,1,4,2,3))
        propagated_images = holo.PropagateToPlaneTF(result, z)
        images = tf.reshape( images, (1, images.shape[0], images.shape[1], images.shape[2],images.shape[3]))
        images = tf.tile(images,( model.network_depth, 1,1,1,1))
        images = tf.transpose( images, (0,1,4,2,3))
        #print(images)
        #print(propagated_images)
        loss = tf.reduce_mean(tf.math.abs(propagated_images) - tf.math.abs(images))
    gradients = g.gradient(loss, model.weights)
    optimizer.apply_gradients(zip(gradients, model.weights))
    # print(pred_y.shape)
    return loss, result, holo.PropagateToPlaneTF(result, z), images
