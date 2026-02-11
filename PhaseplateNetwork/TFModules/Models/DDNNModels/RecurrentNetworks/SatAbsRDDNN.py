from PhaseplateNetwork.TFModules.Models.DDNNModels.RecurrentDDNN import RecurrentDDNN
from PhaseplateNetwork.TFModules.Models.DDNNModels.JonesNetworks.RecurrentSatAbs import RecurrentSatAbs_v1
from PhaseplateNetwork.TFModules.Models.DDNNModels.JonesNetworks.PlateBlock import PlateBlock

class SatAbsRDDNN(RecurrentDDNN):
    def __init__(self, optical_density = 1.0, Isat = 0.1, num_iter = 4, num_blocks = 4, trainable_pixel = 112, plate_scale_factor = 2, propagation_size = 0.001792, propagation_pixel = 224, padding = 0.00175, frequency = 3.843e14, wavespeed = 3e8,**kwargs):

        super(SatAbsRDDNN,self).__init__(NetworkModelForward = None, NetworkModelBackward = None, num_iter = num_iter, trainable_pixel=trainable_pixel, plate_scale_factor= plate_scale_factor, propagation_size = propagation_size, propagation_pixel = propagation_pixel, padding = padding, frequency = frequency, wavespeed =wavespeed,**kwargs)
        self.forward_diffraction = RecurrentSatAbs_v1(optical_density = optical_density, Isat = Isat,block_num = num_blocks, trainable_polarization= 'xy',trainable_pixel = trainable_pixel, plate_scale_factor = plate_scale_factor, propagation_size = propagation_size,
                                                      propagation_pixel = propagation_pixel, padding = padding, frequency = frequency, wavespeed= wavespeed)
        self.backward_diffraction = PlateBlock(trainable_polarization = 'y', block_num = num_blocks, distance = 0.08, trainable_pixel= trainable_pixel, plate_scale_factor = plate_scale_factor, propagation_size = propagation_size,
                                               propagation_pixel = propagation_pixel, padding = padding, frequency = frequency, wavespeed = wavespeed)
        self.append_networks()
        #PlateBlock()
