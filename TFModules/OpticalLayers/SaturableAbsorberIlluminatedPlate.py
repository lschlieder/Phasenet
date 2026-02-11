import tensorflow as tf
from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

def sat_abs_fun(x_in, x_add, I_sat = 0.1, OD = 2.0):
    # amp = saturable_absorber_fn(tf.math.abs(x))
    amp = tf.math.abs(x_in) * tf.math.exp( -(OD /2 ) /( 1 +(tf.math.abs(x_in)**2 +tf.math.abs(x_add)**2) /I_sat))
    phase = tf.math.exp(1j* tf.cast(tf.math.angle(x_in), dtype = tf.complex64))
    return tf.cast(amp ,dtype = tf.complex64) * phase




class SaturableAbsorberIlluminatedPlate(OpticalLayer):
    def __init__(self, N=[100,100], I_in = 1.0, OD = 1.0, I_saturation = 0.1):
        self.inputs = inputs
        self.N = N
        self.I_in = 1.0
        self.OD = OD
        self.I_saturation = 0.1

    def get_intensity_image_from_input(self, input_image, modulation_image, I_sat, OD):
        assert (input_image.shape[0] == illumination_image.shape[0])
        assert (input_image.shape[1] == illumination_image.shape[1])
        assert (input_image.shape[2] == illumination_image.shape[2])
        assert (input_image.shape[3] == illumination_image.shape[3])

        projection = tf.expand_dims(tf.math.reduce_mean(tf.math.abs(modulation_image), axis = 1),axis = 1)

        return sat_abs_fun(input_image, projection, I_sat, OD)

    def call(self,input):
        assert( input.shape[1] == self.N[0])
        assert( input.shape[2] == self.N[1])

        input_image = tf.ones([input.shape[0],N[0], N[1], 1]) * self.I_in
        output_image = get_intensity_image_from_input(input_image, input, self.I_saturation, self.OD)
        return output_image



