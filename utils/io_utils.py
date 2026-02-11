from PhaseplateNetwork.utils.data_utils import get_seperate_input_anim_dataset, get_and_anim_dataset, get_inverse_MNIST_dataset, \
    get_nonlinear_MNIST_dataset, get_numpy_dataset, get_logic_dataset, get_two_input_xy_set, \
    get_two_input_interpolate_set, get_four_input_interpolate_set, get_inverse_measured_MNIST_dataset,\
    get_inverse_measured_MNIST_v2_dataset, get_anim_dataset, get_MNIST_dataset, get_activation_images_from_number, \
    get_fashion_MNIST_dataset, get_logical_dataset, get_activation_images, get_kMNIST_dataset, \
    get_cifar10_dataset, get_and_anim_6inp_dataset, get_inverse_MNIST_thanasi_dataset, \
    get_spiral_dataset, get_SR_latch_dataset, get_MNIST_dataset_3D, get_thanasi_measured_dataset_focus, \
    get_inverse_measured_MNIST_dataset_single, get_mnist_images, get_kmnist_images, get_cifar10_images, get_fashion_mnist_images, \
    get_minerva_optimization_dataset, get_regression_dataset, get_impossible_cube_optimization_dataset, get_impossible_triangle_optimization_dataset
import tensorflow as tf
import numpy as np
from PhaseplateNetwork.Losses.Losses import mse, mse_cropped, normalized_mse, weight_ortogonalization, complex_mse, \
    normalized_complex_mse, cropped_normalized_complex_mse, cropped_complex_mse, mse_loss_per_image_standardized, \
    mse_only_standardized, image_softmax_loss, image_mse_loss, image_detector_mse, image_detector_normalized_mse
from PhaseplateNetwork.Losses.Metrics.AccuracyFromImages import AccuracyFromImages
from PhaseplateNetwork.Losses.Metrics.SoftmaxCrossentropyFromImages import SoftmaxCrossentropyFromImages
from PhaseplateNetwork.Losses.Metrics.SKImage_SSIM import SKImage_SSIM
from PhaseplateNetwork.Losses.Metrics.Tensorflow_SSIM import Tensorflow_SSIM
from PhaseplateNetwork.Losses.Metrics.SNR_Metric import SNR_Metric
import PhaseplateNetwork.Losses.Metrics as Metrics
import argparse
from PhaseplateNetwork.utils.data_utils import get_regression_function

from PhaseplateNetwork.utils.logic_functions import full_adder, SR_latch, multiple_and_function, xor, full_adder_4bit, full_adder_nbit, SR_latch_multi

import matplotlib.pyplot as plt
import PhaseplateNetwork.TFModules.Models.DDNNModels as DDNNModels
from PhaseplateNetwork.TFModules.Models.DiffractiveDeepNeuralNetwork import DDNN
import importlib
import inspect
from keras.callbacks import TensorBoard
from PhaseplateNetwork.utils.Callbacks.TensorboardDDNNCallback import TensorboardDDNNCallback
from PhaseplateNetwork.utils.Callbacks.DDNN_Callbacks import DDNNTrainCallback
from PhaseplateNetwork.utils.Callbacks.TensorboardDDNNCategoricalCallback import TensorboardDDNNCategoricalCallback
from PhaseplateNetwork.utils.Callbacks.TensorboardRegressionDDNNCallback import TensorboardRegressionDDNNCallback

###############Input utils
def get_standard_inputs():
    res = {
        'z' :np.array([0.02, 0.03, 0.04, 0.05]),
        # z = tf.constant([40,90.0])
        'L' : 0.05,
        'N' : 60,
        'padding' : 0.05,
        'r' : 0.025,
        'batch_num':  32,
        'depth' : 5,
        'layer_type' : 'small_res',
        'layer_type' : 'conv',
        'f0': 1.00e6,
        'cM' :1484,
        'epochs' : 10,
        'Network_type' : 'R',
        'PATH' : '/is/ei/lschlieder/Documents/Test_small_res',
        'mnist_scale' : 1,
        'optim_str' : 'admm',
        'learning_rate' : 0.000005,
        'gan' : False,
        'alpha' : 1.0,
        'regularizer' : None,
        'regularization_constant' : 0.0001,
        'loss_function_string' : 'NMSE',
        'activation_function' : 'none'
    }
    return res

def get_standard_DDNN_inputs():
    res = {
        'z': '0.05,0.05,0.05,0.05',
        'L': 0.001792,
        'N': 112,
        's': 2,
        'P': 224,
        'padding': 0.001792,
        'shuffle_num': 1,
        'training':True,
        'batch_size': 32,
        'epochs': 60,
        'output_path':'./',
        'test_path':'./',
        'train_on_mse': 'True',
        'use_FT': True,
        'jitter': False,
        'f0': 3.843e14,
        'cM': 3e8,
        'dataset':'mnist',
        'learning_rate':0.001,
        'optical_density':1.0,
        'saturation_intensity':0.1,
        'optimizer':'adam',
        'save':False,
        'test':False,
        'loss_function':'MSE',
        'metrics':'',
        'output_function':'output_images',
        'callbacks': 'Tensorboard,TensorboardDDNNCallback,DDNNTrainCallback'
    }
    return res

def get_standard_LRGBN_inputs():
    res = {
        'L': 0.05,
        'N': 100,
        'padding': 0.05,
        'output_path': 'Test_Runs/LRGBN_Output',
        'epochs': 60,
        'batch_size': 32,
        'learning_rate': 0.01,
        'f0':  3.843e14,
        'cM':  3e8,
        'dataset':  'mnist',
        'optimizer': 'adam',
        'loss_function': 'MSE',
        'planes': 4

    }
    return res

def get_linear_comparison_inputs():
    res = {
        'model' : 'LinearModel',
        'epochs' : 60,
        'learning_rate': 0.001,
        'dataset': 'mnist',
        'loss_function': 'MSE',
        'optimizer': 'adam',
        'output_path': 'Test_Runs/default_output',
        'batch_size': 32,
        'propagation_pixels': 100
    }
    return res

def parse_arguments_LRGBN():
    options = get_standard_LRGBN_inputs()
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices= ['network_file', 'network_class'], help ='Choose the network to be trained. Either json network file or network_class')
    parser.add_argument("network_config", action ='store',
                            help = 'Depending on mode either json file or network class name')
    parser.add_argument("-o", "--output", action = "store", help = "Speficy the output path",
                            default = options['output_path'])
    parser.add_argument("-L", "--propagation_size", action='store', type=float,
                        help='Size of the input hologram plate in [m]')
    parser.add_argument("-N", "--trainable_pixel", action='store', type=int, help='Pixel number for the hologram plates')
    parser.add_argument("-p", "--padding", action='store', type=float, help='Padding size in [m]')
    parser.add_argument("-b", "--batch_size", action='store', type=int, help='Batch number for training', default = options['batch_size'])
    parser.add_argument("-e", "--epochs", action='store', type=int, help='Epochs for training', default = options['epochs'])
    parser.add_argument("-f", "--frequency", action='store', type=float, help="Wave frequency in [1/s]")
    parser.add_argument("-c", "--wavespeed", action='store', type=float, help="Wavespeed in medium in [mm/s]")
    parser.add_argument("--optimizer", action='store', choices=['adam', 'sgd'],
                    help="Optimizer (\'adam\', \'sgd\')", default=options['optimizer'])
    parser.add_argument("--learning_rate", action='store', type=float, help="Learning rate for training",
                    default=options['learning_rate'])
    parser.add_argument("-d", "--dataset", action='store',
                    choices=['mnist', 'kmnist', 'fashion', 'cifar10'], help='Training dataset',
                    default=options['dataset'])
    parser.add_argument("--planes", action = 'store', type = int)
    parser.add_argument('-z', '--planes_distance', action = 'store')
    parser.add_argument("-E", "--lossfn", action='store',
                    choices=['NMSE', 'SMSE', 'MSE', 'DSSIM', 'ACC', 'XENT', 'DMSE', 'DNMSE'],
                    help='Loss function for network training ( \'NMSE\', \'MSE\', \'SMSE\', \'DSSIM\', \'ACC\', \'XENT\', \'DMSE\')',
                    default=options['loss_function'])
    parser.add_argument("-M", "--metrics", action='store', choices=['SSIM','TFSSIM','SNR', 'MSE', 'SMSE'],
                        help='Metrics to save alogside the loss', nargs='*')


    parser.add_argument("--eagerly", action="store_true", help="Executes the model in tf.eagerly mode (slow).",
                    default=False)
    parser.add_argument("-S", "--save", action='store', help='Load and save the network instead of training it')
    parser.add_argument("-T", "--test", action='store',
                        help='Load and test the network with the test dataset instead of training it')

    parser.add_argument("--images_output_steps", action="store", type = int,
                        help="Change the output frequency. Output after every [arg] images", default=100)
    parser.add_argument("--checkpoints_output_steps", action = "store", type = int,
                        help= "Change the checkpoint output frequency. Output after avery [arg] images", default= 100)
    parser.add_argument("--loss_output_steps", action = "store", type = int,
                        help = "Change the loss output frequency. Output after every [arg] images", default = 100)

    known_args = parser.parse_known_args()



    return known_args[0], known_args[1]


def parse_arguments_linear_model():
    options = get_linear_comparison_inputs()
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--optical_network", action = "store", help= "The network class name of the network to be trained", default = options['model'])
    parser.add_argument("-o", "--output", action='store', help='Specify the output path',
                        default=options['output_path'])
    parser.add_argument("-e", "--epochs", action='store', type=int, help='Epochs for training', default = options['epochs'])
    parser.add_argument("-p", "--propagation_pixel", action='store', type=int, help='Propagation pixel number')
    parser.add_argument("--optimizer", action='store', choices=['adam', 'sgd', 'lfbgs'],
                        help="Optimizer (\'adam\', \'sgd\', \'lfbgs\')", default=options['optimizer'])
    parser.add_argument("--learning_rate", action='store', type=float, help="Learning rate for training",
                        default=options['learning_rate'])
    parser.add_argument("-b", "--batch_size", action='store', type=int, help='Batch number for training',
                        default=options['batch_size'])
    parser.add_argument("-d", "--dataset", action='store',
                    choices=['mnist', 'kmnist', 'cifar10', 'inv_measured_mnist', 'inv_mnist', 'inv_mnist_thanasi','inv_mnist_thanasi_v3',
                             'inv_measured_mnist_v3_single',
                             'inv_measured_v3_focus',
                             'snake_anim','snake_anim_ten_input', 'fish_anim',
                             'fish_anim_ten_input', 'pong_anim', 'focus_change', 'and_anim', 'logic_xor',
                             'logic_SRLatch',
                             'focus_change_2d', 'logic_8and', 'vortex_beam_1234', 'vortex_beam_pm12',
                             'fashion_mnist', 'logic_full_adder', 'logic_4bit_full_adder',
                             'logic_6bit_full_adder', 'and_anim_6inp', 'logic_SRLatch_8','inv_measured_mnist_v2',
                             'and_anim_6inp_measured', 'spiral_dataset','recurrent_SR_latch','inv_measured_mnist_v3',
                             'and_anim_6inp_measured_v2', 'and_anim_4inp_measured_v2', 'minerva_single'], help='Training dataset',
                    default=options['dataset'])
    parser.add_argument("-E", "--lossfn", action='store', choices=['NMSE', 'SMSE', 'MSE', 'DSSIM', 'ACC', 'XENT', 'DMSE', 'DNMSE'],
                    help='Loss function for network training ( \'NMSE\', \'MSE\', \'SMSE\', \'DSSIM\', \'ACC\', \'XENT\', \'DMSE\')',
                    default="MSE")
    parser.add_argument("--eagerly", action = "store_true", help = "Executes the model in tf.eagerly mode (slow).", default = False)
    parser.add_argument("--cpu", action = "store_true", help = "Executes the model on the cpu (very slow)", default = False)
    parser.add_argument("--verbosity", action="store", choices=[0, 1, 2], type=int, default=1)
    parser.add_argument("-M", "--metrics", action='store', choices=['ACC', 'XENT','SSIM','TFSSIM','SNR', 'MSE', 'SMSE'],
                        help='Metrics to save alogside the loss', nargs='*')
    parser.add_argument("-S", "--save", action='store', help='Load and save the network instead of training it')
    parser.add_argument("-T", "--test", action='store',
                        help='Load and test the network with the test dataset instead of training it')
    parser.add_argument("--images_output_steps", action="store", type = int,
                        help="Change the output frequency. Output after every [arg] images", default=100)
    parser.add_argument("--loss_output_steps", action = "store", type = int, help="Change the loss output frequency.", default = 100 )
    known_args = parser.parse_known_args()
    return known_args[0], known_args[1]

    
    
def parse_arguments():
    options = get_standard_DDNN_inputs()

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=['network_str', 'network_file','network_class'], help= "Choose weather to load the network structure from a network str like \"0.03,0.03,0.03\" or a netwok json file, which gives a lot more options or a python class")
    parser.add_argument("network_config", action='store',
                        help='Depending on mode either the network string (i.e. \"0.03,0.03,0.03\" or a text file with a json configuration or a class name')
    parser.add_argument("-o", "--output", action='store', help='Specify the output path',
                        default=options['output_path'])
    parser.add_argument("-L", "--propagation_size", action='store', type=float,
                        help='Size of the input hologram plate in [m]')
    parser.add_argument("-N", "--trainable_pixel", action='store', type=int, help='Pixel number for the hologram plates')
    parser.add_argument("-P", "--propagation_pixel", action='store', type=int, help='Propagation pixel number')
    parser.add_argument("-s", "--plate_scale_factor", action='store', type=int,
                        help='upscaling factor of the plate (only positive integers)')
    parser.add_argument("-p", "--padding", action='store', type=float, help='Padding size in [m]')
    parser.add_argument("-b", "--batch_size", action='store', type=int, help='Batch number for training',
                        default=options['batch_size'])
    parser.add_argument("-e", "--epochs", action='store', type=int, help='Epochs for training', default = options['epochs'])
    parser.add_argument("-f", "--frequency", action='store', type=float, help="Wave frequency in [1/s]")
    parser.add_argument("-c", "--wavespeed", action='store', type=float, help="Wavespeed in medium in [mm/s]")
    parser.add_argument("--optimizer", action='store', choices=['adam', 'sgd', 'lfbgs'],
                        help="Optimizer (\'adam\', \'sgd\', \'lfbgs\')", default=options['optimizer'])
    parser.add_argument("--learning_rate", action='store', type=float, help="Learning rate for training",
                        default=options['learning_rate'])
    parser.add_argument("-d", "--dataset", action='store',
                        choices=['mnist', 'kmnist', 'cifar10', 'inv_measured_mnist', 'inv_mnist', 'inv_mnist_thanasi','inv_mnist_thanasi_v3',
                                 'inv_measured_mnist_v3_single',
                                 'inv_measured_v3_focus',
                                 'snake_anim','snake_anim_ten_input', 'fish_anim',
                                 'fish_anim_ten_input', 'pong_anim', 'focus_change', 'and_anim', 'logic_xor',
                                 'logic_SRLatch',
                                 'focus_change_2d', 'logic_8and', 'vortex_beam_1234', 'vortex_beam_pm12',
                                 'fashion_mnist', 'logic_full_adder', 'logic_4bit_full_adder',
                                 'logic_6bit_full_adder', 'and_anim_6inp', 'logic_SRLatch_8','inv_measured_mnist_v2',
                                 'and_anim_6inp_measured', 'spiral_dataset','recurrent_SR_latch','inv_measured_mnist_v3',
                                 'and_anim_6inp_measured_v2', 'and_anim_4inp_measured_v2', 'minerva_single', 'impossible_cube_single',
                                 'impossible_triangle_single', 'regression_sin',
                                 'regression_x2', 'regression_steps', 'regression_sin2x', 'regression_sin3x', 'regression_sin4x',
                                 'regression_sin_x2', 'regression_x2exp','regression_linear', 'regression_neg_linear',
                                 'regression_const', 'regression_sin_big', 'regression_big_sin2x', 'regression_big_sin3x', 'regression_big_sin4x',
                                 'regression_neg_sin','regression_gaussian','regression_big_gaussian','regression_step','regression_small_step',
                                 'regression_single_step',
                                 'mnist_cat', 'kmnist_cat', 'cifar10_cat', 'fashion_mnist_cat'], help='Training dataset',
                        default=options['dataset'])
    parser.add_argument("--callbacks", action = "store", help = "List with the callbacks ('TB', 'TB-Images', 'FileOutput','TB-Images-cat')", 
                        choices = ['TB', 'TB-Images', 'TB-Images-cat', 'FileOutput', 'ModelCheckpoint', 'BackupAndRestore', 'TB-Regression', 'LearningRateSchedule'], nargs='*')
    parser.add_argument("--number_mnist_digit", action="store", help ="The MNIST digit to train if dataset if inv_measured_v3_focus", type = int)
    parser.add_argument("--jitter", action='store_true', help='Phaseplate jitter')
    parser.add_argument("-E", "--lossfn", action='store', choices=['NMSE', 'SMSE', 'MSE', 'DSSIM', 'ACC', 'XENT', 'DMSE', 'CATXENT', 'DNMSE'],
                        help='Loss function for network training ( \'NMSE\', \'MSE\', \'SMSE\', \'DSSIM\', \'ACC\', \'XENT\', \'DMSE\')',
                        default=options['loss_function'])
    parser.add_argument("-M", "--metrics", action='store', choices=['ACC', 'XENT','SSIM','TFSSIM','SNR', 'MSE', 'SMSE', 'CATACC'],
                        help='Metrics to save alogside the loss', nargs='*')
    parser.add_argument("-S", "--save", action='store', help='Load and save the network instead of training it')
    parser.add_argument("-T", "--test", action='store',
                        help='Load and test the network with the test dataset instead of training it')
    parser.add_argument("--optical_density", action='store', type=float,
                        help="Optical density for the saturable absorber", default=options['optical_density'])
    parser.add_argument("--saturation_intensity", action='store', type=float,
                        help="saturation intensity for the saturable absorber plates")
    parser.add_argument("--output_preprocessing", action='store', choices=['waveprop'],
                        help="filter the images of the output dataset with the given function (waveprop filters for possible creatable wavelengths")
    parser.add_argument("--output_intensity_scaling", action='store', type=float,
                        help="Output intensity scaling with given number", default=1.0)
    parser.add_argument("--use_matrix_fourier_transform", action='store_true',
                        help='Use the discrete fast fourier transform for wave propagation instead of matrix multiplication')
    parser.add_argument("--images_output_steps", action="store", type = int,
                        help="Change the output frequency. Output after every [arg] images", default=100)
    parser.add_argument("--loss_output_steps", action = "store", type = int, help="Change the loss output frequency.", default = 100 )
    parser.add_argument("--lense_focus", action="store", type=float, help="Change the focus of the lenses in the DDNN",
                        default=0.1)
    parser.add_argument("--verbosity", action="store", choices=[0, 1, 2], type=int, default=1)
    parser.add_argument("--phase_input", action ="store_true", help = 'Change the input images to a phase input' )
    parser.add_argument("--eagerly", action = "store_true", help = "Executes the model in tf.eagerly mode (slow).", default = False)
    parser.add_argument("--cpu", action = "store_true", help = "Executes the model on the cpu (very slow)", default = False)
    parser.add_argument("-g" "--data_size", action = "store", help = "size of the input data", default = None)
    known_args = parser.parse_known_args()
    #args = parser.parse_args()
    #print(args)
    #print(known_args[1])
    #input()

    return known_args[0], known_args[1]

def parse_arguments_DDON():
    options = get_standard_DDNN_inputs()

    parser = argparse.ArgumentParser()
    #parser.add_argument("mode", choices=['network_str', 'network_file','network_class'], help= "Choose weather to load the network structure from a network str like \"0.03,0.03,0.03\" or a netwok json file, which gives a lot more options or a python class")
    #parser.add_argument("network_config", action='store',
    #                    help='Depending on mode either the network string (i.e. \"0.03,0.03,0.03\" or a text file with a json configuration or a class name')
    parser.add_argument("-o", "--output", action='store', help='Specify the output path',
                        default=options['output_path'])

    parser.add_argument("--optical_network", action = 'store', type = str, help ='Class name of the optical network')
    parser.add_argument("-b", "--batch_size", action='store', type=int, help='Batch number for training',
                        default=options['batch_size'])
    parser.add_argument("-e", "--epochs", action='store', type=int, help='Epochs for training', default = options['epochs'])


    parser.add_argument("--optimizer", action='store', choices=['adam', 'sgd', 'lfbgs'],
                        help="Optimizer (\'adam\', \'sgd\', \'lfbgs\')", default=options['optimizer'])
    parser.add_argument("--learning_rate", action='store', type=float, help="Learning rate for training",
                        default=options['learning_rate'])

    parser.add_argument("-d", "--dataset", action='store',
                        choices=['mnist', 'kmnist','fashion_mnist', 'cifar10'], help='Training dataset',
                        default=options['dataset'])

    parser.add_argument('--image_size', action = 'store', type = int, default = 30, help = 'image size')
    parser.add_argument("-E", "--lossfn", action='store', choices=['NMSE', 'SMSE', 'MSE', 'DSSIM'],
                        help='Loss function for network training ( \'NMSE\', \'MSE\', \'SMSE\', \'DSSIM\')',
                        default=options['loss_function'])
    parser.add_argument("-M", "--metrics", action='store', choices=['SSIM','TFSSIM','SNR', 'MSE', 'SMSE'],
                        help='Metrics to save alogside the loss', nargs='*')
    parser.add_argument("-S", "--save", action='store', help='Load and save the network instead of training it')
    parser.add_argument("-T", "--test", action='store',
                        help='Load and test the network with the test dataset instead of training it')
    parser.add_argument("--use_matrix_fourier_transform", action='store_true',
                        help='Use the discrete fast fourier transform for wave propagation instead of matrix multiplication')
    parser.add_argument("--images_output_steps", action="store", type = int,
                        help="Change the output frequency. Output after every [arg] images", default=100)
    parser.add_argument("--loss_output_steps", action='store', default = 100, help = "number of batches after which to save the loss")
    parser.add_argument("--checkpoint_output_steps", action='store', default = 100, help = "number of batches after which to save the checkpoints")
    parser.add_argument("--phase_input", action ="store_true", help = 'Change the input images to a phase input' )
    parser.add_argument("--eagerly", action = "store_true", help = "Executes the model in tf.eagerly mode (slow).", default = False)
    parser.add_argument("--timesteps", action = "store", default = 100, type = int, help = "number of denoising timesteps for the scheduler")
    parser.add_argument("--beta_start", action = "store", default = 0.001,type = float, help ="beta start for the denoiser")
    parser.add_argument("--beta_end", action = "store", default = 0.01,type = float, help = "beta end for densoing")
    parser.add_argument("--predict_noise", action ='store_true', help = "If classified train directly on noise")
    known_args = parser.parse_known_args()
    #args = parser.parse_args()
    #print(args)
    #print(known_args[1])
    #input()

    return known_args[0], known_args[1]


def output_preprocessing(dataset, function):
    def fun(x,y):
        #x,y = set
        return x, function(y)
    new_set = dataset.map(fun)
    return new_set

def input_preprocessing(dataset, function):
    def fun(x,y):
        return function(x), y
    new_set = dataset.map(fun)
    return new_set

def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c

def get_model(args, unkown_args, class_prefix = 'PhaseplateNetwork.TFModules.Models.DDNNModels.'):
    def get_kwargs(args, keywords):
        #params =  {'trainable_pixel':args.pixel_number, 'scaling': args.plate_scale_factor, 'L': args.propagation_size, 'propagation_pixel': args.propagation_pixel,'padding':args.padding, 'f0':args.frequency, 'cM':args.wavespeed}
        kwargs = {}
        for key in keywords.parameters:
            #print(key)
            value = getattr(args, key, None)
            if not value == None:
                kwargs[key] = value
        return kwargs

    def get_unkown_args(kwargs, ukargs, keywords):
        for key in keywords.parameters:
            keytype = type(keywords.parameters[key].default)
            for i in range(0, len(ukargs),2):
                if key == ukargs[i][2:]:
                    value = keytype(ukargs[i+1])
                    kwargs[key] = value
        return kwargs


    if args.mode == 'network_str':
        keywords = inspect.signature(DDNN.from_network_str)
        kwargs = get_kwargs(args,keywords)
        model = DDNN.from_network_str(args.network_config, **kwargs)
    elif args.mode =='network_file':
        try:
            with open(args.network_config) as f:
                #print(f.read())
                #print(json.load(f))
                data = f.read()
        except:
            raise SystemExit('Could not find input json file {}...... Exiting!'.format(args.network_config))
        #model = DDNN.from_config(data)
        model = tf.keras.models.model_from_json(data, custom_objects = {'DDNN':DDNN})

    elif args.mode =='network_class':
        #try:
        network_name = args.network_config.split('.')[-1]
        model_class = class_for_name(class_prefix+args.network_config, network_name)
        keywords = inspect.signature(model_class.__init__)
        kwargs = get_kwargs(args, keywords)
        kwargs = get_unkown_args(kwargs,unkown_args,keywords)
        model = model_class( **kwargs )
        #except:
            #raise SystemExit('Could not find model class {}...... Exiting!'.format('DDNNModels.'+args.network_config))
    return model

def get_optical_model(args, unkown_args, class_prefix = 'PhaseplateNetwork.TFModules.Models.DDNNModels.'):
            #try:
    def get_kwargs(args, keywords):
        #params =  {'trainable_pixel':args.pixel_number, 'scaling': args.plate_scale_factor, 'L': args.propagation_size, 'propagation_pixel': args.propagation_pixel,'padding':args.padding, 'f0':args.frequency, 'cM':args.wavespeed}
        kwargs = {}
        for key in keywords.parameters:
            #print(key)
            value = getattr(args, key, None)
            if not value == None:
                kwargs[key] = value
        return kwargs

    def get_unkown_args(kwargs, ukargs, keywords):
        for key in keywords.parameters:
            keytype = type(keywords.parameters[key].default)
            for i in range(0, len(ukargs),2):
                if key == ukargs[i][2:]:
                    value = keytype(ukargs[i+1])
                    kwargs[key] = value
        return kwargs

    #network_name = args.network_config.split('.')[-1]
    network_name = args.optical_network.split('.')[-1]
    model_class = class_for_name(class_prefix+args.optical_network, network_name)
    keywords = inspect.signature(model_class.__init__)
    kwargs = get_kwargs(args, keywords)
    kwargs = get_unkown_args(kwargs,unkown_args,keywords)
    model = model_class( **kwargs )
        #except:
            #raise SystemExit('Could not find model class {}...... Exiting!'.format('DDNNModels.'+args.network_config))
    return model

def get_linear_model(args, unkown_args, class_prefix = 'PhaseplateNetwork.TFModules.Models.ClassificationModels.'):
            #try:
    def get_kwargs(args, keywords):
        #params =  {'trainable_pixel':args.pixel_number, 'scaling': args.plate_scale_factor, 'L': args.propagation_size, 'propagation_pixel': args.propagation_pixel,'padding':args.padding, 'f0':args.frequency, 'cM':args.wavespeed}
        kwargs = {}
        for key in keywords.parameters:
            #print(key)
            value = getattr(args, key, None)
            if not value == None:
                kwargs[key] = value
        return kwargs

    def get_unkown_args(kwargs, ukargs, keywords):
        for key in keywords.parameters:
            keytype = type(keywords.parameters[key].default)
            for i in range(0, len(ukargs),2):
                if key == ukargs[i][2:]:
                    value = keytype(ukargs[i+1])
                    kwargs[key] = value
        return kwargs

    #network_name = args.network_config.split('.')[-1]
    network_name = args.optical_network.split('.')[-1]
    model_class = class_for_name(class_prefix+args.optical_network, network_name)
    keywords = inspect.signature(model_class.__init__)
    kwargs = get_kwargs(args, keywords)
    kwargs = get_unkown_args(kwargs,unkown_args,keywords)
    model = model_class( batch_num = args.batch_size, pixel = args.propagation_pixel, **kwargs )
        #except:
            #raise SystemExit('Could not find model class {}...... Exiting!'.format('DDNNModels.'+args.network_config))
    return model



def get_hologram_dataset_from_string(dataset_str, batch_num = 10, N = 60, output_planes = 4 ):
    if dataset_str == 'mnist':
        scale = (N//28)
        train_dataset, test_dataset = get_MNIST_dataset_3D(batch_num, N, output_planes, 'mnist', scale)
    if dataset_str == 'kmnist':
        scale = (N//28)
        train_dataset, test_dataset = get_MNIST_dataset_3D(batch_num, N, output_planes,'kmnist', scale)
    if dataset_str == 'fashion':
        scale = (N//28)
        train_dataset, test_dataset = get_MNIST_dataset_3D(batch_num, N, output_planes,'fashion', scale)
    if dataset_str == 'cifar10':
        scale = (N//32)
        train_dataset, test_dataset = get_MNIST_dataset_3D(batch_num, N, output_planes, 'cifar10', scale)
    return train_dataset, test_dataset

def get_dataset_from_string(dataset_str, batch_num = 10, N= 60, L = 50, scale = 1, phase_input = False, preprocessing_fn = lambda x: x, mnist_num = None, **kwargs):
    '''
    get a train and test dataset from a string. If no test dataset is available, test dataset is train dataset.
    params:
    dataset_str: the dataset string. 
    batch_num: batch_number
    N: size of the images
    L: Size of the images in [m]
    scale: scale of eventuall output images
    phase_input: if applicable, phase output (Better to do this with encodings)
    '''
    print(dataset_str)
    if dataset_str == 'mnist':
        mnist_scale = (N // 28)
        result_detector_radius = L/10
        train_dataset, test_dataset = get_MNIST_dataset(batch_num, int(N), L, N, scale, mnist_scale,
                                                        result_detector_radius,
                                                        train_on_mse=True)

    elif dataset_str =='fashion_mnist':
        mnist_scale = (N//28)
        result_detector_radius = L/10
        train_dataset, test_dataset = get_fashion_MNIST_dataset(batch_num, int(N), L, N, scale, mnist_scale,
                                                        result_detector_radius,
                                                        train_on_mse = True)

    elif dataset_str == 'kmnist':
        mnist_scale = (N//28)
        result_datector_radius = L/10
        train_dataset, test_dataset = get_kMNIST_dataset(batch_num, int(N), L, N, scale, mnist_scale,
                                                         result_datector_radius,
                                                         train_on_mse = True)

    elif dataset_str == 'cifar10':
        #pic_scale = (N // 32)
        pic_scale = N/32
        #pic_scale = 1
        result_detector_radius = L / 10
        train_dataset, test_dataset = get_cifar10_dataset(batch_num, N, L, pic_scale,
                                                         result_detector_radius,
                                                         train_on_mse=True)

    elif dataset_str == 'inv_mnist':
        mnist_scale = (N // 28)
        result_detector_radius = L/10
        train_dataset, test_dataset = get_inverse_MNIST_dataset(batch_num, int(N), L, N, scale, mnist_scale,
                                                  result_detector_radius)

    elif dataset_str == 'inv_mnist_thanasi':
        mnist_scale = (N//28)
        result_detector_radius = 0.004
        train_dataset, test_dataset = get_inverse_MNIST_thanasi_dataset(batch_num, int(N), L, N, scale, mnist_scale, result_detector_radius)

    elif dataset_str =='inv_mnist_thanasi_v3':
        mnist_scale = (N//28)
        train_dataset, test_dataset = get_inverse_MNIST_thanasi_dataset(batch_num,int(N), L,N, scale, mnist_scale, version = 2)

    elif dataset_str == 'inv_measured_v3_focus':
        mnist_scale = (N//28)
        train_dataset, test_dataset = get_thanasi_measured_dataset_focus(batch_num, N, L,L/20,  version =3)
    elif dataset_str == 'inv_measured_mnist':
        mnist_scale = (N // 28)
        result_detector_radius = 4
        train_dataset,test_dataset = get_inverse_measured_MNIST_dataset(batch_num, int(N * scale) + padding, L, N, scale,
                                                           mnist_scale, version = 1)
    elif dataset_str == 'inv_measured_mnist_v2':
        mnist_scale = (N // 28)
        train_dataset, test_dataset= get_inverse_measured_MNIST_dataset(batch_num, int(N), L, scale, mnist_scale,version =2)
        #test_dataset = train_dataset
    elif dataset_str == 'inv_measured_mnist_v3':
        mnist_scale = (N//28)
        train_dataset,test_dataset = get_inverse_measured_MNIST_dataset(batch_num, int(N),L, scale, mnist_scale, version = 3)
        #test_dataset = train_dataset

    elif dataset_str == 'inv_measured_mnist_v3_single':
        mnist_scale = (N//28)
        train_dataset, test_dataset = get_inverse_measured_MNIST_dataset_single(int(N), mnist_scale, version = 3, mnist_num = mnist_num )

    elif dataset_str == 'focus_change':
        train_dataset = get_two_input_interpolate_set(10000, batch_num, int(N * scale) + padding, L, N, scale, radius=4,
                                                      radius_out=1)
        test_dataset = train_dataset
    elif dataset_str == 'focus_change_2d':
        train_dataset = get_four_input_interpolate_set(10000, batch_num, N , L, N,1,
                                                       radius=L/10,
                                                       radius_out=L/20)
        test_dataset = train_dataset
    elif dataset_str == 'logic_xor':
        train_dataset = get_logic_dataset(batch_num, 2, xor, N, L, radius=L/10,
                                          input_type='circles', double=True)
        #train_dataset = train_dataset.skip(1)
        #test_dataset = get_logic_dataset(batch_num, 3, full_adder, N* scale + padding,L,N, scale, radius = 0.4, input_type='circles', double = True)
        test_dataset = train_dataset
    elif dataset_str == 'logic_full_adder':
        train_dataset = get_logic_dataset(batch_num, 3, full_adder, N, L, radius=L/10,
                                          input_type='circles', double=True)
        #(batch_num, input_num, logical_function, image_size, L, radius = 4, input_type = 'circles', double = False):
        #test_dataset = get_logic_dataset(batch_num, 3, full_adder, N* scale + padding,L,N, scale, radius = 0.4, input_type='circles', double = True)
        test_dataset = train_dataset

    elif dataset_str == 'logic_4bit_full_adder':
        train_dataset = get_logic_dataset(batch_num, 8, full_adder_4bit, N, L, radius = L/10, input_type = 'circles', double = True)
        test_dataset = train_dataset

    elif dataset_str == 'logic_6bit_full_adder':
        fun = lambda x: full_adder_nbit(x,6)
        train_dataset = get_logic_dataset(batch_num, 12, fun, N, L, radius = L/20, input_type = 'circles', double = True)
        test_dataset = train_dataset

    elif dataset_str == 'logic_SRLatch':
        train_dataset = get_logic_dataset(batch_num, 2, SR_latch, N, L, N, scale, radius=L/10,
                                          input_type='circles', double=True)
        test_dataset = train_dataset

    elif dataset_str == 'logic_SRLatch_8':
        train_dataset = get_logic_dataset(batch_num, 8, SR_latch_multi, N, L, radius = L/10, input_type='circles', double = True )
        test_dataset = train_dataset

    elif dataset_str == 'logic_8and':

        train_dataset = get_logic_dataset(batch_num, 8, multiple_and_function, N, L,
                                          radius=L/10,
                                          input_type='circles', double=True)
        test_dataset = train_dataset
    elif dataset_str == 'and_anim':
        train_dataset = get_and_anim_dataset(batch_num, N, L, N, radius=L/10,
                                             input_type='circles',
                                             double=True)
        test_dataset = train_dataset

    elif dataset_str == 'and_anim_6inp':
        train_dataset = get_and_anim_6inp_dataset(batch_num, N, L, radius=L/10,
                                             input_type='circles')
        test_dataset = train_dataset

    elif dataset_str == 'and_anim_6inp_measured':
        train_dataset = get_and_anim_6inp_dataset(batch_num, N, L,
                                             input_type='measured')
        test_dataset = train_dataset

    elif dataset_str == 'and_anim_6inp_measured_v2':
        train_dataset,test_dataset = get_and_anim_6inp_dataset(batch_num, N, L,
                                             input_type='measured_v2')
    elif dataset_str == 'and_anim_4_inp_measured_v2':
        train_dataset, test_dataset = get_and_anim_dataset(batch_num, N,L,N, radius=L/10, input_type = 'measured_v2')


    elif dataset_str == 'snake_anim':
        train_dataset = get_anim_dataset(batch_num, int(N * scale) + padding, L, N, scale,
                                         anim_path='./snake_amin_30px.npy')
        test_dataset = train_dataset
    elif dataset_str == 'fish_anim':
        train_dataset = get_anim_dataset(batch_num, int(N * scale) + padding, L, N, scale,
                                         anim_path='./fish_anim_30px.npy')
        test_dataset = train_dataset
    # elif dataset =='fish_anim_ten_input':
    #    train_dataset = get_seperate_input_anim_dataset( batch_num, int(N*scale)+padding, L , N, scale, anim_path ='./fish_anim_30px.npy'))
    elif dataset_str == 'snake_anim_ten_input':
        train_dataset = get_seperate_input_anim_dataset(batch_num, int(N * scale) + padding, L, N, scale,
                                                        anim_path='./snake_amin_30px.npy')
        test_dataset = train_dataset
    elif dataset_str == 'pong_anim' and N == 30:
        train_dataset = get_anim_dataset(batch_num, int(N * scale) + padding, L, N, scale, anim_path='./pong.npy')
        test_dataset = train_dataset
    elif dataset_str == 'pong_anim' and N == 60:
        train_dataset = get_anim_dataset(batch_num, int(N * scale) + padding, L, N, scale, anim_path='./pong_60px.npy')
        tests_dataset = train_dataset
    elif dataset_str == 'vortex_beam_1234' and N == 30:
        train_dataset = get_numpy_dataset(4, int(N), L, N, 1,
                                          input_path='./VortexBeam/30px/bessel_vortex_1234/inputs.npy',
                                          output_path='./VortexBeam/30px/bessel_vortex_1234/outputs.npy')
        test_dataset = train_dataset
        # loss_fn = cropped_normalized_complex_mse
    elif dataset_str == 'vortex_beam_1234' and N == 60:
        train_dataset = get_numpy_dataset(4, int(N), L, N, 1,
                                          input_path='./VortexBeam/60px/bessel_vortex_1234/inputs.npy',
                                          output_path='./VortexBeam/60px/bessel_vortex_1234/outputs.npy')
        test_dataset = train_dataset
        # loss_fn = cropped_normalized_complex_mse
    elif dataset_str == 'vortex_beam_pm12' and N == 30:
        train_dataset = get_numpy_dataset(4, int(N), L, N, 1,
                                          input_path='./VortexBeam/30px/bessel_vortex_pm12/inputs.npy',
                                          output_path='./VortexBeam/30px/bessel_vortex_pm12/outputs.npy')
        test_dataset = train_dataset
        # loss_fn = cropped_normalized_complex_mse
    elif dataset_str == 'vortex_beam_pm12' and N == 60:
        train_dataset = get_numpy_dataset(4, int(N), L, N, 1,
                                          input_path='./VortexBeam/60px/bessel_vortex_pm12/inputs.npy',
                                          output_path='./VortexBeam/60px/bessel_vortex_pm12/outputs.npy')
        test_dataset = train_dataset
    elif dataset_str == 'spiral_dataset':
        train_dataset,test_dataset = get_spiral_dataset(batch_num,(N,N), L,L/10)
    elif dataset_str == 'recurrent_SR_latch':
        train_dataset = get_SR_latch_dataset(batch_num, N, L)
        test_dataset = train_dataset
    elif dataset_str == 'minerva_single':
        train_dataset, test_dataset = get_minerva_optimization_dataset(batch_num, N, L, **kwargs)

    elif dataset_str == 'impossible_cube_single':
        train_dataset, test_dataset = get_impossible_cube_optimization_dataset(batch_num, N, L, **kwargs)

    elif dataset_str == 'impossible_triangle_single':
        train_dataset, test_dataset = get_impossible_triangle_optimization_dataset(batch_num, N, L, **kwargs)

    elif dataset_str == 'mnist_cat':
        if N == None:
            N = 28
        mnist_scale = (N // 28)
        result_detector_radius = L/10
        train_dataset, test_dataset = get_MNIST_dataset(batch_num, int(N), L, N, scale, mnist_scale,
                                                        result_detector_radius,
                                                        train_on_mse=False)
    elif dataset_str =='fashion_mnist_cat':
        if N == None:
            N = 28
        mnist_scale = (N//28)
        result_detector_radius = L/10
        train_dataset, test_dataset = get_fashion_MNIST_dataset(batch_num, int(N), L, N, scale, mnist_scale,
                                                        result_detector_radius,
                                                        train_on_mse = False)

    elif dataset_str == 'kmnist_cat':
        if N == None:
            N = 28
        mnist_scale = (N//28)
        result_datector_radius = L/10
        train_dataset, test_dataset = get_kMNIST_dataset(batch_num, int(N), L, N, scale, mnist_scale,
                                                         result_datector_radius,
                                                         train_on_mse = False)

    elif dataset_str == 'cifar10_cat':
        if N == None:
            N = 32
        #pic_scale = (N // 32)
        pic_scale = N/32
        #pic_scale = 1
        result_detector_radius = L / 10
        train_dataset, test_dataset = get_cifar10_dataset(batch_num, N, L, pic_scale,
                                                         result_detector_radius,
                                                         train_on_mse=False)

    

    elif dataset_str in ['regression_sin','regression_x2', 'regression_steps', 'regression_sin2x',
                          'regression_sin_x2', 'regression_sin3x', 'regression_sin4x', 'regression_x2exp',
                          'regression_linear', 'regression_neg_linear', 'regression_const', 'regression_sin_big', 
                          'regression_neg_sin', 'regression_big_sin4x', 'regression_big_sin2x', 'regression_big_sin3x',
                          'regression_gaussian','regression_big_gaussian','regression_step','regression_small_step',
                          'regression_single_step', 'regression_relu']:

        train_dataset, test_dataset = get_regression_dataset( batch_num, 10000, dataset_str)
    else:
        raise ValueError("Unkown dataset: {}".format(dataset_str))

    test_dataset = output_preprocessing(test_dataset, preprocessing_fn)
    train_dataset = output_preprocessing(train_dataset, preprocessing_fn)

    if phase_input:
        def phase_input_fn(x):
            return tf.cast( tf.ones_like(x),dtype=tf.complex64)* tf.math.exp(1j* tf.cast((x*2*np.pi - 0.5), dtype = tf.complex64))
        train_dataset = input_preprocessing(train_dataset, phase_input_fn)
        test_dataset =  input_preprocessing(test_dataset, phase_input_fn)



    return train_dataset, test_dataset

def get_image_dataset_from_string( dataset_name = 'mnist', image_size = 30, scale = 1):
    if dataset_name == 'mnist':
        train_images, test_images = get_mnist_images(30,1, True)
    elif dataset_name == 'kmnist':
        train_images, test_images = get_kmnist_images(image_size, scale, True)
    elif dataset_name == 'fashion_mnist':
        train_images, test_images = get_fashion_mnist_images(image_size, scale, True)
    elif dataset_name == 'cifar10':
        train_images, test_images = get_cifar10_images(image_size, scale, True)
    return train_images, test_images



def get_loss_function_from_string( loss_function_string, N= 50, L = 60, dataset_str =None):
    if loss_function_string == 'MSE':
        loss_fn = mse
    elif loss_function_string == 'NMSE':
        loss_fn = normalized_mse
    elif loss_function_string == 'DSSIM':
        loss_fn = dssim
    elif loss_function_string == 'ACC':
        loss_fn = acc
    elif loss_function_string == 'SMSE':
        loss_fn = mse_loss_per_image_standardized
    elif loss_function_string == 'OSMSE':
        loss_fn = mse_only_standardized
    elif loss_function_string == 'CMSE':
        loss_fn = complex_mse
    elif loss_function_string == 'CNCMSE':
        loss_fn = cropped_normalized_complex_mse
    elif loss_function_string == 'CCMSE':
        loss_fn = cropped_complex_mse

    elif loss_function_string == 'CATXENT':
        loss_fn = tf.keras.losses.CategoricalCrossentropy( from_logits = True)

    elif loss_function_string == 'XENT':
        if dataset_str == 'spiral_dataset':
            activation_imgs = get_activation_images_from_number(4,N,L,L/10)
        else:
            activation_imgs = get_activation_images_from_number(10, N, L, L/10)
        loss_fn = lambda wanted_image, imgs: image_softmax_loss(imgs, wanted_image, activation_imgs)

    elif loss_function_string == 'DMSE':
        if dataset_str == 'spiral_dataset':
            activation_imgs = get_activation_images_from_number(4,N,L,L/10)
        else:
            activation_imgs = get_activation_images_from_number(10, N, L , L/10)
        output_area = np.expand_dims(np.sum(activation_imgs, axis = 0),axis = 0)
        loss_fn = lambda wanted_image,imgs: image_detector_mse(imgs,wanted_image, output_area)
    elif loss_function_string == 'DNMSE':
        if dataset_str == 'spiral_dataset':
            activation_imgs = get_activation_images_from_number(4, N, L, L / 10)
        else:
            activation_imgs = get_activation_images_from_number(10, N, L, L / 10)
        output_area = np.expand_dims(np.sum(activation_imgs, axis=0), axis=0)
        loss_fn = lambda wanted_image, imgs: image_detector_normalized_mse(imgs, wanted_image, output_area)
    #elif loss_function_string == 'IMSE':
    #    loss_fn = lambda output_image, wanted_vec: image_mse_loss(output_image, wanted_vec, L, 4)
    #    output_function = lambda result, wanted_output, path, i: output_vector(result, wanted_output, path,
    #                                                                           activation_imgs, i)
    return loss_fn

def get_optimizer(optimizer, learning_rate = None):
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    else:
        print('Error: Optimizer {} not found'.format(optimizer))
    return opt

def get_callbacks_from_string(callback_string = ''):
    callbacks = callback_string.split(',')
    for c in callbacks:
        cback_class = class_for_name("PhaseplateNetwork.utils.Callbacks"+c, network_name)


def get_metrics(metrics, metric_str, dataset_str, N = 60, L = 50):

    if metric_str == 'SSIM':
        #if dataset_str in ['inverse_mnist', 'inverse_measured_mnist']:
        metrics.append( SKImage_SSIM(name = 'SSIM'))
    if metric_str == 'TFSSIM':
        metrics.append( Tensorflow_SSIM(name ='TFSSIM'))

    if metric_str == 'SNR':
        metrics.append( SNR_Metric(name = 'SNR'))

    if metric_str == 'SMSE':
        metrics.append( Metrics.SMSE_Metric(name = 'SMSE'))

    if metric_str == 'MSE':
        metrics.append(Metrics.MSE_Metric(name ='MSE'))

    if metric_str == 'CATACC':
        metrics.append(tf.keras.metrics.CategoricalAccuracy(name = 'Categorical Accuracy'))

    if metric_str == 'ACC':
        if dataset_str in  ['mnist', 'fashion_mnist', 'kmnist', 'cifar10']:
            #print(N)
            #print(L)
            #print(L/10)
            class_imgs = get_activation_images_from_number(10, N, L, L/10)
            #print(class_imgs.shape)
            metrics.append(  AccuracyFromImages(class_imgs, name='Accuracy') )
        elif dataset_str in ['spiral_dataset']:
            class_imgs = get_activation_images_from_number(4,N,L, L/10)
            metrics.append( AccuracyFromImages(class_imgs, name='Accuracy'))

        if dataset_str in ['logic_8and', 'logic_xor','logic_SRLatch_8']:
            if dataset_str == 'logic_8and':
                log_fun = multiple_and_function
                inp_num = data_size
            if dataset_str == 'logic_xor':
                log_fun = full_adder
                inp_num = 3
            if dataset_str == 'logic_SRLatch_8':
                log_fun = SR_latch_multi
                inp_num = 8

            input_logical, output_logical = get_logical_dataset(inp_num, log_fun, True)
            activation_imgs = np.expand_dims(get_activation_images(input_logical, int(N), L,L/10),3).astype('float32')
            print(activation_imgs.shape)
            #for i in range(0,activation_imgs.shape[0]):
                #print(activation_imgs.shape)
                #plt.imshow(activation_imgs[i,:,:])
                #plt.show()
            metrics.append( AccuracyFromImages( activation_imgs, name='Accuracy'))
            #activation_imgs = get_activation_images(10,N, L, L/10)

    if metric_str == 'XENT':
        if dataset_str in ['mnist', 'fashion_mnist', 'cifar10', 'kmnist']:
            #class_imgs
            activation_imgs = get_activation_images_from_number(10, N, L, L/10)
            metrics.append(  SoftmaxCrossentropyFromImages(  activation_imgs, name= 'Crossentropy'))
            #loss_fn = lambda imgs, wanted_image: image_softmax_loss(imgs, wanted_image, activation_imgs)
        elif dataset_str in ['spiral_dataset']:
            class_imgs = get_activation_images_from_number(4,N,L, L/10)
            metrics.append( SoftmaxCrossentropyFromImages(class_imgs, name='Crossentropy'))
        if dataset_str in ['logic_8and', 'logic_xor']:
            if dataset_str == 'logic_8and':
                log_fun = multiple_and_function
                inp_num = 8
            if dataset_str == 'logic_xor':
                log_fun = full_adder
                inp_num = 3

            input_logical, output_logical = get_logical_dataset(inp_num, log_fun, True)
            activation_imgs = get_activation_images(input_logical, int(N), L,L/10)
            metrics.append( SoftmaxCrossentropyFromImages( activation_imgs, name = 'Crossentropy'))
            #activation_imgs = get_activation_images(10,N, L, L/10)

    return metrics


def get_training_callbacks(args, example_data):
    callbacks = []
    if args.callbacks != None:
        for callback in args.callbacks:
            if callback == 'TB':
                tensorboard = TensorBoard(log_dir=args.output+'/tensorboard_logs', write_graph=True, write_images=True)
                callbacks.append(tensorboard)
            elif callback == 'TB-Images':
                image_callback = TensorboardDDNNCallback(example_data, PATH = args.output+'/tensorboard_logs', images_output_steps = args.images_output_steps, loss_output_steps= args.loss_output_steps)
                callbacks.append(image_callback)
            elif callback == 'FileOutput':
                DDNN_files = DDNNTrainCallback(output_function, example_data, args.output, args.images_output_steps)
                callbacks.append(DDNN_files)
            elif callback == 'TB-Images-cat':
                image_cat_callback = TensorboardDDNNCategoricalCallback(example_data, PATH = args.output+'/tensorboard_logs', images_output_steps = args.images_output_steps, loss_output_steps= args.loss_output_steps)
                callbacks.append(image_cat_callback)
            elif callback == 'ModelCheckpoint':
                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.output+'/chkpts/ckpt_{epoch:02d}',
                                                    save_weights_only=True,
                                                    verbose=1)
                callbacks.append(cp_callback)
            elif callback == 'TB-Regression':  
                tensorboard_regression = TensorboardRegressionDDNNCallback(get_regression_function(args.dataset),  PATH = args.output+'/tensorboard_logs',  images_output_steps = args.images_output_steps,loss_output_steps= args.loss_output_steps )

            elif callback == 'BackupAndRestore':
                restore_callback = tf.keras.callbacks.BackupAndRestore(args.output+'/backup_dir',
                                                                    save_freq='epoch',
                                                                    delete_checkpoint=True)
                callbacks.append(restore_callback)
            elif callback == 'LearningRateSchedule':
                def scheduler(epoch, lr):
                    if epoch < 10:
                        return lr     
                    else:     
                        return lr * tf.math.exp(-0.2)
                lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
                callbacks.append(lr_callback)


    return callbacks













