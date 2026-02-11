import tensorflow as tf
from PhaseplateNetwork.TFModules.ForwardOperators.ForwardOperator import ForwardOperator
from PhaseplateNetwork.TFModules.ForwardOperators.WavePropagationForwardOperator import WavePropagationForwardOperator



class MultiplateForwardOperator(ForwardOperator):
    def __init__(self, z, trainable_pixel=100, L=80, scale = 2, padding=5, f0=2.00e6, cM=1484, use_FT=True, jitter=False, **kwargs):
        '''
        MultiplateForwardOperator that creates a forward operator, that takes a starting wave distribution and multiple phase shifts inbetween the values z
        :param z: Distances in [mm]. Inbetween the distances z the wave field is phase shifted by given phase plates
        :param trainable_pixel: the number of trainable pixel on the trainable plate
        :param scale: the scale with which the trainable pixel are multiplied with
        :param L: The size of the trainable plate in [mm]
        :param padding: the padding for the propagation in pixels
        :param PhaseTrainable: Set the Phase part of the phaseplate to be trainable
        :param AmplitudeTrainable: Set the amplitude part of the phaseplate to be trainable
        :param f0: Frequency in 1/s
        :param cM: Wavespeed in medium in [mm]/s
        :param use_FT: Use Fourier transform instead or Matrix multiplication for PropagationLayer
        :param use_tanh_activation: Take the tanh of the phase and multiply by pi so that the result is between -pi:pi
        :param jitter: Use a random 1 pixel jitter every propagation so that the network is more robust to experimental errors
        :param kwargs: Pass through arguments

        '''
        super(MultiplateForwardOperator, self).__init__(**kwargs)
        self.N = trainable_pixel
        self.z = z
        self.scale = scale
        self.L = L
        self.padding = padding
        self.f0 = f0
        self.cM = cM
        self.use_FT = use_FT
        self.jitter = jitter
        self.propagations = []
        for distance in z:
            #(self, z, N, L, padding = 50, f0=2.00e6, cM=1484, channels_last=True, use_FT=True, ** kwargs):
            self.propagations.append(WavePropagationForwardOperator(distance, self.N, L,scale,  padding, f0, cM, True, use_FT))

    def call(self, Input, PhaseShifts):
        new_field = Input
        assert( PhaseShifts.shape[0] == len(self.propagations)-1)
        i = tf.constant(0)
        for i in range(0,PhaseShifts.shape[0]):
            PhaseShift = tf.reshape( PhaseShifts[i,:,:], shape= ( 1, PhaseShifts.shape[1], PhaseShifts.shape[2], 1))
            new_field = self.propagations[i].call(new_field) * tf.cast(PhaseShift, dtype = tf.complex64)
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







