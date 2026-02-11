import tensorflow as tf
from .OpticalLayer import OpticalLayer

class IntensityToPhase(OpticalLayer):
    def __init__(self,**kwargs):
        super(IntensityToPhase,self).__init__(**kwargs)

    def call(self,Input):
        amp = tf.ones_like(tf.math.abs(Input))
        phase = 1j*np.pi*2*tf.complex(tf.math.abs(Input)/tf.maximum(Input),0.0)
        out = tf.complex(amp,0.0)* tf.math.exp(phase)
        return out
