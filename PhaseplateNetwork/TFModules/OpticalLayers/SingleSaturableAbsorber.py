import tensorflow as tf

from PhaseplateNetwork.TFModules.OpticalLayers.OpticalLayer import OpticalLayer

class SingleSaturableAbsorber(OpticalLayer):
    def __init__(self,OD = 1.0, I_saturation = 0.1, **kwargs):
        super(SingleSaturableAbsorber, self).__init__(**kwargs)
        self.OD = OD
        self.I_saturation = I_saturation


    def sat_abs_fun(self, x):
        amp = tf.math.abs(x) * tf.math.exp( -(self.OD/2)/(1+tf.math.abs(x)**2/self.I_saturation))
        phase = tf.math.exp(1j* tf.cast(tf.math.angle(x), dtype = tf.complex64))
        return tf.cast(amp,dtype = tf.complex64) * phase

    def call(self, input):
        #amp = tf.math.abs(input)
        #angle = tf.math.angle(input)

        return self.sat_abs_fun(input)


    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        temp = {
            "OpticalDensity": self.OD,
            "SaturationIntensity": self.I_saturation
        }
        return temp

    @classmethod
    def from_config(cls, config):
        el = cls(OD = config['OpticalDensity'], I_saturation = config['SaturationIntensity'])
        return el