import tensorflow as tf
import numpy as np
import PhaseplateNetwork.TFModules.ForwardOperators as fo
from PhaseplateNetwork.utils.util import repeat_image_tensor
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer
from PhaseplateNetwork.TFModules.OpticalLayers.PaddingLayer import PaddingLayer
from PhaseplateNetwork.TFModules.OpticalLayers.CropLayer import CropLayer


class WavePropagationVariableDistance(OpticalLayer):
    '''
    Wave Propagation Layer Class. Used to propagate a wavefield a certain distance via the angular spectrum method.
    '''

    def get_propagation_operator(self, ka, kx, ky, z,L):
        '''
        Get the propagation operator with which the angular spectrum is multiplied
        Ignore evanescent waves.
        :param ka: wavenumber calculated via 2 * np.pi / wavelength
        :param kx: kspace grid ( [mm])
        :param ky: kspace grid ( [mm])
        :param z: Propagation distance [mm]
        :return: e^( i * z * sqrt( ka**2 - kx**2 - ky**2))       ( with antialiasing threshold)
        '''
        kz_real = tf.multiply(ka, ka) - tf.multiply(kx, kx) - tf.multiply(ky, ky)
        is_larger_zero = tf.cast(kz_real > 0, dtype=tf.float32)
        is_larger_zero = tf.cast(tf.multiply(kx, kx) + tf.multiply(ky, ky) < tf.multiply(ka, ka), dtype=tf.float32)
        kz_real = kz_real * is_larger_zero
        z = tf.reshape(tf.cast(z,tf.float32), (z.shape[0],1,1))
        propagator = tf.exp(1j * tf.cast(z * tf.sqrt(kz_real), dtype=tf.complex64))
        D = L
        K_threshold = ka * tf.sqrt((D ** 2 / 2) / ((D ** 2 / 2) + z ** 2))

        is_smaller_threshold = tf.cast((self.kx ** 2 + self.ky ** 2) < K_threshold ** 2, dtype=tf.complex64)

        propagator = propagator * is_smaller_threshold

        return propagator

    # @tf.function
    def get_r(self, i, j, dp):
        '''
        Returns the radius for two input numbers i,j on two NxN grids with distance z
        :param i: index of first pixel on NxN grid (row major)
        :param j: index of the second pixel on NxN grid ( row major)
        :param dp: pixel size in [mm]
        :return: distance from pixel with index i to pixel with index j
        '''
        posix, posiy = self.get_position(i, self.N_in)
        posjx, posjy = self.get_position(j, self.N_out)
        return (np.sqrt((posjx * dp - posix * dp) ** 2 + (posjy * dp - posiy * dp) ** 2 + self.z ** 2))

    def get_position(self, i, N):
        '''
        Returns the x and y position for the pixel with index i on an NxN grid
        :param i: pixel index (row major)
        :param N: Gridsize
        :return: (x,y) - pixel position
        '''
        return i % N, i // N

    def get_index(self, x, y, N):
        '''
        Returns the index i of a pixel on an NxN grid on position (x,y)
        :param x: x position of pixel
        :param y: y position of pixel
        :param N: Gridsize
        :return: index on grid ( row major)
        '''
        return y * N + x



    # @tf.function
    def get_propagation_matrix(self, N_in, N_out, z, dp, wavelength):
        '''
        Returns the propagation matrix when not using the angular spectrum method but the direct calculationg method
        This is not feasable for big inputs since the TFModules stores the complete matrix ( N**2 x N**2) in memory.
        :param N_in: size of the input grip
        :param N_out: size of the output grid
        :param z: distance of the two plates in [mm]
        :param dp: pixel size in [mm]
        :param wavelength: wavelength in [mm]
        :return: The propagation matrix that can be multiplied with the input vector ( Size  N**2x N**2 !!!!)
        '''
        w = np.zeros((N_in * N_in, N_out * N_out), dtype='complex64')
        i, j = np.meshgrid(np.arange(0, N_in * N_in, 1), np.arange(0, N_out * N_out, 1))
        ixpos, iypos = self.get_position(i, N_in)
        jxpos, jypos = self.get_position(j, N_out)
        r = np.sqrt((jxpos * dp - ixpos * dp) ** 2 + (jypos * dp - iypos * dp) ** 2 + self.z ** 2)


        first = tf.complex((z / r ** 2),0.0) * (tf.complex((1 / (2 * np.pi * r) + 1),0.0) / (1j*tf.complex( wavelength,0.0)))
        second = tf.math.exp(1j *tf.complex(2 * np.pi * r / wavelength,0.0))
        w = first *second
        #w = tf.cast(w,dtype = tf.complx64)
        print(w.dtype)
        w_tf = tf.constant(w, dtype=tf.complex64)
        return w_tf

    def __init__(self, N, L, padding = None, f0=2.00e6, cM=1484, channels_last = True, use_FT=True,**kwargs):
        '''
        Create the Propagation Layer
        :param z: propagation distance
        :param N: Pixel number
        :param L: Propagation area size in [m]
        :param padding: The amount of padding on the side of the propagation. If this is zero, wraparound artifacts start to interferre
        :param f0: Frequency in 1/s
        :param cM: Wavespeed in medium in [m]/s
        :param channels_last: Bool if the image channels are last
        :param use_FT: Use the angular spectrum method or the matrix multiplication method
        :param kwargs: Stuff for keras
        '''
        super(WavePropagationVariableDistance, self).__init__(**kwargs)

        # Create padding and cropping layers
        self.N = N
        self.padding = padding
        self.L = L
        if padding != None:
            #print(N)
            #print(L)
            #print(padding)
            self.padding_layer = PaddingLayer(N, L, padding)
            prop_shape = self.padding_layer.compute_output_shape(tf.TensorShape([1, N, N, 1]))
            #print(prop_shape)
            self.N_p = prop_shape[1]
            self.crop_layer = CropLayer((prop_shape[1],prop_shape[2]), L, padding)
            self.pad = True
            self.size = L + 2*padding
        else:
            self.pad = False
            self.N_p = N
            self.size = L



        self.dp = self.size / (self.N_p - 1)
        self.df = 1 / self.size
        self.wavelength = cM / f0
        #print("wavelength: {}".format(self.wavelength))




        self.fmax = 1 / self.wavelength
        self.use_FT = use_FT
        self.channels_last = channels_last

        self.f_print_max = 1/(2*self.dp)


        self.phi_max = np.arcsin(self.wavelength * self.f_print_max)
        self.fully_connected_distance = self.size / (np.tan(self.phi_max))
        #print("Fully connected distance: {}".format(self.fully_connected_distance))


        self.f0 = f0
        self.cM = cM

        ## Create the propagation operator
        if use_FT:
            [self.fx, self.fy] = np.meshgrid((np.arange(-self.N_p / 2, self.N_p / 2, 1)) * self.df,
                                             (np.arange(-self.N_p / 2, self.N_p / 2, 1)) * self.df)
            self.kx = (2 * np.pi * self.fx).astype('float32')
            self.ky = (2 * np.pi * self.fy).astype('float32')


            self.ka = np.float32(2 * np.pi / self.wavelength)


            #self.propagation_operator = self.get_propagation_operator(self.ka, self.kx, self.ky, z, L)
            #self.inverse_propagation_operator = self.get_propagation_operator(self.ka, self.kx, self.ky, -z, L)
        else:
            self.N_in = self.N_p
            self.N_out = self.N_p
            #self.w = self.get_propagation_matrix(self.N_p, self.N_p, z, self.dp, self.wavelength)
            #self.inverse_w = self.get_propagation_matrix(self.N_p, self.N_p, -z, self.dp, self.wavelength)

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.N, self.N,input_shape[3]]
    # @tf.function

    def call(self, input, z):
        '''
        Performs the propagation for a batch of input images
        :param input: images with dimension (batch, N, N, 1) or (batch, 1, N, N) depending on self.channels_last
        :return: the propagated images with input dimensions and len(z) channels (batch,N,N,len(z)) or (batch,len(z),N,N)
        '''
        #if not isinstance(z, np.ndarray):
        #    z = np.array([z])
        z = tf.reshape(tf.cast(z, tf.float32), (len(z), 1, 1))
        img = input
        if self.pad:
            img = self.padding_layer(img)

        if self.channels_last == True:
            img = tf.transpose(img, (0, 3, 1, 2))


        if self.use_FT:

            # calculate the 2d fourier transform of the wave
            fourier_pressure = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(img)))
            # Multiply by kspace propagator
            self.propagation_operator = self.get_propagation_operator(self.ka, self.kx, self.ky, z, self.L)

            propagated_kspace = tf.multiply(fourier_pressure, self.propagation_operator)
            # Retransform in image space
            propagated_image = tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(propagated_kspace)))
        else:
            img = tf.reshape(img, (img.shape[0], img.shape[1], self.N_in * self.N_in, 1))
            self.w = self.get_propagation_matrix(self.N_p, self.N_p, z, self.dp, self.wavelength)
            propagated_image = tf.matmul(self.w, img, True)
            propagated_image = tf.reshape(propagated_image, (img.shape[0], img.shape[1], self.N_out, self.N_out))

        if self.channels_last == True:
            propagated_image = tf.transpose(propagated_image, (0, 2, 3, 1))

        #downscale image again
        #propagated_image = propagated_image[:, ::self.scale,::self.scale,:]
        #unpad
        #propagated_image = tf.image.crop_to_bounding_box(propagated_image, self.padding_pixel // 2, self.padding_pixel // 2, self.N,
        #                                                 self.N)
        if self.pad:
            propagated_image = self.crop_layer(propagated_image)
        return propagated_image

    def inverse_call(self, input,z):
        '''
        Calculates the inverse propagation with -z
        :param input: The pressure field resulting from 'call' with dimensions (batch, N, N, 1) or (batch, 1, N, N) depending on self.channels_last
        :return: The creating pressure field with the same dimensions
        '''
        z = tf.reshape(tf.cast(z, tf.float32), (len(z), 1, 1))
        img = input
        img = tf.image.pad_to_bounding_box(img, self.padding_pixel // 2, self.padding_pixel // 2, self.actual_N, self.actual_N)

        if self.channels_last == True:
            img = tf.transpose(img, (0, 3, 1, 2))

        if self.use_FT:
            # calculate the 2d fourier transform of the wave
            fourier_pressure = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(img))) * (1 / self.N) ** 2
            # propagator = get_propagation_operator(ka, kx, ky, z)
            self.inverse_propagation_operator = self.get_propagation_operator(self.ka, self.kx, self.ky, -z, self.L)
            propagated_kspace = tf.multiply(fourier_pressure, self.inverse_propagation_operator)
            # transform the kspace to image-space
            propagated_image = tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(propagated_kspace))) / (
                    1 / self.N) ** 2

        else:
            img = tf.reshape(img, (img.shape[0], img.shape[1], self.N_in * self.N_in, 1))
            self.inverse_w = self.get_propagation_matrix(self.N_p, self.N_p, -z, self.dp, self.wavelength)
            propagated_image = tf.matmul(self.inverse_w, img, True)
            propagated_image = tf.reshape(propagated_image, (img.shape[0], img.shape[1], self.N_out, self.N_out))

        if self.channels_last == True:
            propagated_image = tf.transpose(propagated_image, (0, 2, 3, 1))

        propagated_image = tf.image.crop_to_bounding_box(propagated_image, self.padding_pixel // 2, self.padding_pixel // 2,
                                                             self.N,
                                                             self.N)
        return propagated_image


    def get_config(self):
        temp ={
            "z": self.z,
            "N": self.N,
            "L": self.L,
            "padding": self.padding,
            "f0": self.f0,
            "cM": self.cM,
            "channels_last": self.channels_last,
            "use_FT": self.use_FT
        }
        return temp
