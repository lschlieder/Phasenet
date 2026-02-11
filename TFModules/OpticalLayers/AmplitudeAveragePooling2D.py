import tensorflow as tf

class AmplitudeAveragePooling2D(tf.keras.layers.AveragePooling2D):
    def __init__(self,*kwargs):
        super(AmplitudeAveragePooling2D,self).__init__(*kwargs)

    def call(self, inputs):
        inp_amp = tf.math.abs(inputs)
        inp_phase = tf.math.angle(inputs)
        out_amp = super(AmplitudeAveragePooling2D,self).call(inp_amp)
        out_phase = super(AmplitudeAveragePooling2D,self).call(inp_phase)

        return tf.cast(out_amp+ 1e-10, dtype = tf.complex64) * tf.math.exp( 1j* tf.cast(out_phase,dtype = tf.complex64))