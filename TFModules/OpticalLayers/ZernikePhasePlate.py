import tensorflow as tf
import numpy as np
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

EPSILON = 1e-16


def R(n, m, rho):
    if (n - m) % 2 == 0:
        top = (n - m) // 2
        sum = 0.0
        for k in range(0, top + 1):
            add = rho ** (n - 2 * k) * ((((-1) ** k) * np.math.factorial(n - k)) / (
                        np.math.factorial(k) * np.math.factorial(top - k) * np.math.factorial(top - k)))
            sum = sum + add
    else:
        sum = 0.0 * rho
    return sum

def tf_factorial(num):
    return tf.math.exp( tf.math.lgamma(num+1))

def R_tf(n,m,rho):
    reminder = tf.math.floormod(n-m,2)

    def even_function(n,m,rho):
        top = tf.math.floordiv((n-m),2)
        sum = 0.0
        for k in range(0, top+1):
            add = tf.math.pow(rho, n-2*k) * ((((-1) ** k) * tf_factorial(n - k)) / (
                        tf_factorial(k) * tf_factorial(top - k) * tf_factorial(top - k)))
            sum = sum + add
    return tf.cond( tf.math.equal(reminder, 0) , even_function(n,m,rho), lambda : rho*0.0)

def ZernikePolynomial_tf(n,m,rho,phi):
    return tf.cond( tf.math.greater_equal(m, 0), lambda : R(n,m,rho)*tf.math.cos(m*phi), lambda : R(n,-m, rho)*tf.math.sin(-m*phi))

def ZernikePolynomial(n, m, rho, phi):
    res = 0.0
    if m >= 0:
        res = R(n, m, rho) * tf.math.cos(m * phi)
    elif m < 0:
        res = R(n, -m, rho) * tf.math.sin(-m * phi)
    return res

def get_ZernikePolynomials(max_n,Nx,Ny):
    polys = []
    x = np.linspace(-1,1,Nx)
    y = -np.linspace(-1,1,Ny)
    x,y = np.meshgrid(x,y)
    r = np.sqrt(x**2+y**2)
    mask = (r < 1.0)
    phi = (np.arctan(y/x)+np.pi)%(np.pi)
    for n in range(0,max_n):
        for m in range(-n,n,2):
            polys.append(tf.cast(tf.constant(ZernikePolynomial(n,m,r,phi)* mask),dtype = tf.float32))
    return polys




class ZernikePhasePlate(OpticalLayer):
    def __init__(self, shape, depth = 12, **kwargs):
        '''
        A simple layer to multiply the incoming complex field with a phase plate that has (possible trainable) amplitude and phase
        '''
        super(ZernikePhasePlate, self).__init__(**kwargs)
        self.phaseplate_shape = [1, shape[0], shape[1],1]
        #self.polys = get_ZernikePolynomials(7, shape[0],shape[1])
        self.depth = depth
        self.number_coefficients = self.depth*(self.depth+1)//2
        #print(len(self.polys))
        self.coefficients = tf.Variable(tf.constant(0.0, shape= (self.number_coefficients), dtype = tf.float32))
        self.amplitudes = tf.reshape(tf.constant(np.ones(self.phaseplate_shape), dtype = tf.float32), [1, shape[0], shape[1], 1])

        x = np.linspace(-1, 1, shape[0])
        y = -np.linspace(-1, 1, shape[1])
        x, y = np.meshgrid(x, y)
        r = np.sqrt(x ** 2 + y ** 2)
        mask = (r < 1.0)
        phi = (np.arctan(y / x) + np.pi) % (np.pi)
        self.r = tf.constant(r,dtype = tf.float32)
        self.phi = tf.constant(phi, dtype = tf.float32)
        self.mask = tf.constant(mask, dtype = tf.float32)
        #self.amplitudes = tf.Variable(np.ones(self.phaseplate_shape),trainable = amplitude_trainable, dtype = tf.float32)
        #self.phases = tf.Variable(np.zeros(self.phaseplate_shape), trainable= phase_trainable,dtype = tf.float32)



    def call(self, Input):
        #assert( Input.shape[1] == self.phases.shape[1])
        #assert( Input.shape[2] == self.phases.shape[2])
        #assert( Input.shape[3] == self.phases.shape[3])
        phases = tf.zeros([Input.shape[1], Input.shape[2]],dtype = tf.float32)
        i = 0

        for n in range(0,self.depth):
            for m in range(-n,n,2):
                phases = phases + self.coefficients[i] * ZernikePolynomial_tf(n,m,self.r, self.phi)*self.mask
                i = i +1

            #phases = phases + self.coefficients[i] * tf.reshape(self.polys[i], ( Input.shape[1], Input.shape[2],1))
        phases = tf.reshape(phases, [1,Input.shape[1],Input.shape[2], 1])
        phase_plate = tf.complex(self.amplitudes, 0.0) * tf.cast(
            tf.exp(1j * tf.cast(phases, dtype=tf.complex64)), dtype=tf.complex64)
        new_field = Input * phase_plate
        return new_field

    def compute_output_shape(self, input_shape):
        #assert( input_shape[1] == self.phases.shape[1])
        #assert( input_shape[2] == self.phases.shape[2])
        #assert( input_shape[3] == self.phases.shape[3])
        output_shape = []
        for i in range(0,len(input_shape)):
            output_shape.append(input_shape[i])
        return output_shape

    def get_image_variables(self):
        phases = tf.zeros([self.phaseplate_shape[1], self.phaseplate_shape[2]],dtype = tf.float32)
        i = 0
        for n in range(0,self.depth):
            for m in range(-n,n,2):
                phases = phases + self.coefficients[i] * ZernikePolynomial_tf(n,m,self.r, self.phi)*self.mask
                i = i +1

            #phases = phases + self.coefficients[i] * tf.reshape(self.polys[i], ( Input.shape[1], Input.shape[2],1))
        #phases = tf.reshape(phases, [1,self.phaseplate_shape[1], self.phaseplate_shape[2], 1])
        return phases
