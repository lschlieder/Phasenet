import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PhaseplateNetwork.TFModules.PropagationLayer import PropagationLayer

epsilon = 1e-10

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
        self.dense1 = tf.keras.layers.Dense(60 * 60)
        self.flatten = tf.keras.layers.Flatten()

        self.conv_out = tf.keras.layers.Conv2D(1, 3, padding='same')

    def call(self, inputs):
        out = self.dense1(self.flatten(self.batch_normalization_out(self.conv2(self.batch_normaliztion2(self.conv1(self.batch_normalization1(inputs)))))))
        out = self.conv_out(tf.reshape(out, (-1, 60,60, 1))) + tf.expand_dims(tf.math.reduce_mean(inputs,axis = 3), axis = 3)
        #out = self.conv_out(self.batch_normalization_out(
        #    self.conv2(self.batch_normaliztion2(self.conv1(self.batch_normalization1(inputs)))))) + tf.expand_dims(
            #tf.math.reduce_mean(inputs, axis=3), axis=3)

        # u_new_c = tf.expand_dims(tf.math.reduce_mean(proj, axis=3), axis=3)

        return out


class ResNetNoRepeatModel(tf.keras.Model):


    def __init__(self, z, L, N, transducer_radius=25, padding=50, depth=20, f0=2.00e6, cM=1484, use_IASA=False,
                 layer_type='conv'):
        '''
        Create The Residual Hologram Network model
        :param z :
        '''
        super(ResNetNoRepeatModel, self).__init__()

        self.L = L
        self.N = N
        self.padding = padding
        self.z = z
        self.use_IASA = use_IASA
        self.layer_type = layer_type
        dx = L / N
        X, Y = np.mgrid[-L / 2:L / 2:dx, -L / 2:L / 2:dx]
        amplitude = np.zeros_like(X)
        amplitude[X ** 2 + Y ** 2 < transducer_radius ** 2] = 1.0
        self.amplitude = np.reshape(amplitude, (amplitude.shape[0], amplitude.shape[1], 1))

        self.conv_blocks = []
        for i in range(0,depth):
            if layer_type == 'conv':
                self.conv_blocks.append( ProjectionBlockConv())
            elif layer_type == 'dense':
                self.conv_blocks.append( ProjectionBlockDense())
            else:
                print('Layer type {} not found'.format(layer_type))
                raise RuntimeError('Layer type not found')




        self.z = tf.cast(z, dtype=tf.float32)
        self.network_depth = depth
        self.propagation_layers = []
        for z_i in z:
            self.propagation_layers.append(PropagationLayer(z_i, N, L, padding, f0, cM))






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

    # @tf.function
    def call(self, inputs, **kwargs):
        img = inputs

        u = tf.constant(np.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2], 1)))
        u = (self.amplitude + epsilon) * tf.exp(1j * tf.cast(tf.math.angle(u), dtype=tf.complex64))
        u_array = []

        u_res = u

        for i in range(0, self.network_depth):
            proj = self.projection_to_image_constraint(u, img)
            #proj = complex_to_channel(proj)
            #print(proj.shape)
            proj = tf.concat((tf.math.angle(proj),img, tf.math.angle(u)), axis = 3)
            out = self.conv_blocks[i].call(proj)
            #print(out.shape)
            #out = channel_to_complex(out)
            #print(out.shape)
            out = (self.amplitude + epsilon) * tf.exp(1j * tf.cast(out, dtype=tf.complex64))
            u = self.project_to_transducer_constraint(out)

            u_array.append(tf.math.angle(u))

        #result = tf.stack(u_array)

        prop_u = []
        for layer in self.propagation_layers:
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
    return loss, result, holo.PropagateToPlaneTF(result, z), images
