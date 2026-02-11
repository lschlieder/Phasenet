import tensorflow as tf
from PhaseplateNetwork.TFModules.ForwardOperators.ForwardOperator import ForwardOperator
from PhaseplateNetwork.TFModules.ForwardOperators.WavePropagation import WavePropagation



class PhasePlateForwardOperator(ForwardOperator):
    def __init__(self, trainable_pixel=100, L=80, scale = 2, f0=2.00e6, cM=1484,**kwargs):
        '''
        A simple layer to multiply the incoming complex field with a phase plate that has (possible trainable) amplitude and phase
        '''
        super(PhasePlateForwardOperator, self).__init__(**kwargs)
        self.N = trainable_pixel
        self.scale = scale
        self.L = L
        self.f0 = f0
        self.cM = cM



    def call(self, Input, PhaseShifts):
        new_field = Input
        assert( PhaseShifts.shape[0] == len(self.propagations)-1)
        i = tf.constant(0)
        for i in range(0,PhaseShifts.shape[0]):
            PhaseShift = tf.reshape( PhaseShifts[i,:,:], shape= ( 1, PhaseShifts.shape[1], PhaseShifts.shape[2], 1))
            new_field = self.propagations[i].call(new_field) * tf.exp(1j * tf.cast(PhaseShift, dtype = tf.complex64))
            #new_field = new_field * tf.exp(1j * tf.cast(PhaseShifts[i,:,:], dtype=tf.complex64))
        new_field = self.propagations[len(self.propagations)-1].call(new_field)
        return new_field

    def inverse_call(self,Input, PhaseShifts):
        new_field = Input
        assert( Phaseshifts.shape[0] == len(self.propagations)-1)
        for i in range(0, len(PhaseShift.shape[0])):
            new_field = self.propagations[i].inverse_call(new_field)*tf.exp(-1j * tf.cast(PhaseShifts[i,:,:], dtype = tf.complex64))


        new_field = self.propagations[len(propagations)-1].inverse_call(new_field)
        return new_field

    def get_phaseshift_shape(self):
        return tf.TensorShape([len(self.propagations)-1,self.N, self.N])







