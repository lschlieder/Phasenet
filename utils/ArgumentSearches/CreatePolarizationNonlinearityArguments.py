import os 
import numpy as np


def create_files_and_arguments_name(plates, plates_per_rotation, use_hologram, nonlinear, nonlinear_mode, trainable_pixel, propagation_size, dataset, losses):
    file_str = []
    arguments = []
    for plate_num in plates:
        for size in propagation_size:
            for d in dataset:
                for hologram in use_hologram:
                    for p in trainable_pixel:
                        for l in losses:
                            for nl in nonlinear:
                                for ppr in plates_per_rotation:
                                    if plate_num < ppr:
                                        print('plate num is {} and plates per rotation is {}. breaking, since this would not create nonlinearity'. format(plate_num, ppr))
                                        break

                                    hologram_str = "--use_holoram True" if hologram else ""
                                    #nonlinear_str = "--nonlinearity True" if nl else ""
                                    if not nl:
                                        if ppr == 1:
                                            file =  f"{d}_depth:{plate_num}_ppr_{ppr}_hologram_{hologram}_nonlinear:{nl}_nlmode:0_N:{p}_L:{size}_loss:{l}"
                                            argument_str= f"network_class JonesNetworks.PolarizationNonlinearity -o /is/cluster/work/lschlieder/DDNN/PolarizationNonlinearity/{d}/{file} -N {p} -s 1 -P {p} -L {size} -d {d} -M ACC XENT MSE SMSE -E {l} --learning_rate 0.1 --images_output_steps 1000 -e 60 -b 32 --plate_per_nonlinearity {ppr} --plates {plate_num} {hologram_str}\n"
                                            file_str.append(file)
                                            arguments.append(argument_str)
                                        else:
                                            print('ppr is not 1, but nonlinear is True. Breaking so that only one linear file for this configuration exists')
                                            break

                                    else:
                                        for nlmode in nonlinear_mode:
                                            file =  f"{d}_depth:{plate_num}_ppr_{ppr}_hologram_{hologram}_nonlinear:{nl}_nlmode:{nlmode}_N:{p}_L:{size}_loss:{l}"
                                            argument_str= f"network_class JonesNetworks.PolarizationNonlinearity -o /is/cluster/work/lschlieder/DDNN/PolarizationNonlinearity/{d}/{file} -N {p} -s 1 -P {p} -L {size} -d {d} -M ACC XENT MSE SMSE -E {l} --learning_rate 0.1 --nonlinearity True --images_output_steps 1000 -e 60 -b 32 --plate_per_nonlinearity {ppr} --plates {plate_num} {hologram_str} --nonlinear_mode {nlmode}\n"

                                            
                                            #file =  "{}_depth:{}_ppr_{}_hologram_{}_nonlinear:{}_nlmode:{}_N:{}_L:{}_loss:{}".format(d, plate_num, ppr, hologram, nl,nlmode, p,size,l)
                                            #argument_str = "network_class JonesNetworks.PolarizationNonlinearity -o /is/cluster/work/lschlieder/DDNN/PolarizationNonlinearity/{}/{} -N {} -s 1 -P {} -d {} -M ACC XENT MSE SMSE -E {} --learning_rate 0.1 --nonlinearity True --images_output_steps 1000 -e 60 -b 32\n".format(d, file,p, p,d, l)
                                            file_str.append(file)
                                            arguments.append(argument_str)

                         
                                    
    return file_str, arguments

arguments_mnist_output = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/JonesPropagation/PolarizationNonlinearity/Mnist.txt"
arguments_mnist_output_16 = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/JonesPropagation/PolarizationNonlinearity/Mnist_16.txt"
arguments_mnist_output_248 = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/JonesPropagation/PolarizationNonlinearity/Mnist_248.txt"

arguments_kmnist_output = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/JonesPropagation/PolarizationNonlinearity/Kmnist.txt"
arguments_kmnist_output_16 = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/JonesPropagation/PolarizationNonlinearity/Kmnist_16.txt"
arguments_kmnist_output_248 = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/JonesPropagation/PolarizationNonlinearity/Kmnist_248.txt"

arguments_cifar10_output = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/JonesPropagation/PolarizationNonlinearity/Cifar10.txt"
arguments_cifar10_output_16 = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/JonesPropagation/PolarizationNonlinearity/Cifar10_16.txt"
arguments_cifar10_output_248 = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/JonesPropagation/PolarizationNonlinearity/Cifar10_248.txt"

plates = [2,4,8,16]
plates_per_rotation = [1,2,4]
use_hologram = [False]
nonlinear = [True, False]
nonlinear_mode = ['same','ortho']
trainable_pixel = [120]
propagation_size = [0.001792]
dataset = ['mnist', 'kmnist', 'cifar10']
dataset_mnist = ['mnist']
dataset_kmnist = ['kmnist']
dataset_cifar10 = ['cifar10']
losses = ['MSE', 'SMSE', 'XENT']

files_mnist, arguments_mnist = create_files_and_arguments_name(plates, plates_per_rotation, use_hologram, nonlinear, nonlinear_mode, trainable_pixel, propagation_size, dataset_mnist, losses)
files_mnist_16, arguments_mnist_16 = create_files_and_arguments_name([16], plates_per_rotation, use_hologram, nonlinear, nonlinear_mode, trainable_pixel, propagation_size, dataset_mnist, losses)
files_mnist_248, arguments_mnist_248 = create_files_and_arguments_name([2,4,8], plates_per_rotation, use_hologram, nonlinear, nonlinear_mode, trainable_pixel, propagation_size, dataset_mnist, losses)

files_kmnist, arguments_kmnist = create_files_and_arguments_name(plates, plates_per_rotation, use_hologram, nonlinear, nonlinear_mode, trainable_pixel, propagation_size, dataset_kmnist, losses)
files_kmnist_16, arguments_kmnist_16 = create_files_and_arguments_name([16], plates_per_rotation, use_hologram, nonlinear, nonlinear_mode, trainable_pixel, propagation_size, dataset_kmnist, losses)
files_kmnist_248, arguments_kmnist_248 = create_files_and_arguments_name([2,4,8], plates_per_rotation, use_hologram, nonlinear, nonlinear_mode, trainable_pixel, propagation_size, dataset_kmnist, losses)


files_cifar10, arguments_cifar10 = create_files_and_arguments_name(plates, plates_per_rotation, use_hologram, nonlinear, nonlinear_mode, trainable_pixel, propagation_size, dataset_cifar10, losses)
files_cifar10_16, arguments_cifar10_16 = create_files_and_arguments_name([16], plates_per_rotation, use_hologram, nonlinear, nonlinear_mode, trainable_pixel, propagation_size, dataset_cifar10, losses)
files_cifar10_248, arguments_cifar10_248 = create_files_and_arguments_name([2,4,8], plates_per_rotation, use_hologram, nonlinear, nonlinear_mode, trainable_pixel, propagation_size, dataset_cifar10, losses)


#files_mnist_16, arguments_mnist_16 = create_files_and_arguments_name([16], plates_per_rotation, use_hologram, nonlinear, nonlinear_mode, trainable_pixel, propagation_size, dataset_mnist, losses)
#files_mnist_842, arguments_mnist_842 = create_files_and_arguments_name([2,4,8], plates_per_rotation, use_hologram, nonlinear, nonlinear_mode, trainable_pixel, propagation_size, dataset_mnist, losses)
#files_mnist_4, arguments_mnist_4 = create_files_and_arguments_name([16], plates_per_rotation, use_hologram, nonlinear, nonlinear_mode, trainable_pixel, propagation_size, dataset_mnist, losses)
#files_mnist_2, arguments_mnist_2 = create_files_and_arguments_name([8], plates_per_rotation, use_hologram, nonlinear, nonlinear_mode, trainable_pixel, propagation_size, dataset_mnist, losses)



#
# print(len(files_mnist))

with open(arguments_mnist_output, 'w') as f:
    f.writelines(arguments_mnist)
with open(arguments_mnist_output_16, 'w') as f:
    f.writelines(arguments_mnist_16)
with open(arguments_mnist_output_248, 'w') as f:
    f.writelines(arguments_mnist_248) 

with open(arguments_kmnist_output, 'w') as f:
    f.writelines(arguments_kmnist)
with open(arguments_kmnist_output_16, 'w') as f:
    f.writelines(arguments_kmnist_16)
with open(arguments_kmnist_output_248, 'w') as f:
    f.writelines(arguments_kmnist_248)

with open(arguments_cifar10_output, 'w') as f:
    f.writelines(arguments_cifar10)
with open(arguments_cifar10_output_16, 'w') as f:
    f.writelines(arguments_cifar10_16)
with open(arguments_cifar10_output_248, 'w') as f:
    f.writelines(arguments_cifar10_248)


