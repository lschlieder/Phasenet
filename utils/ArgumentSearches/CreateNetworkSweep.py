import os 
import numpy as np

def get_folder_name( datasets = ['mnist'],
distances = [0.05],
normalized = [False],
phase = [False],
plate_num = [8],
plate_scale_factor = [1],
abs_nonlinear = [False],
loss_fn = ['MSE']):
    strings= []
    for dataset in datasets:
        if dataset == 'mnist':
            dataset_path = 'MNIST'
        elif dataset == 'kmnist':
            dataset_path = 'KMNIST'
        elif dataset == 'cifar10':
            dataset_path = 'CIFAR10'
        print(dataset_path)

        
        for distance in distances:
            for normal_network in normalized:
                if normal_network:
                    network_str = "DiffractionPlateNetworkPoolingNormalized"
                else:
                    network_str = "DiffractionPlateNetworkPooling"

                for phase_inp in phase:
                    if phase_inp:
                        phase_str = "--phase_encoding 1"
                    else:
                        phase_str = ""

                    for plate_number in plate_num:
                        for psf in plate_scale_factor:
                            for l in loss_fn:
                                folder_name = "/{}/D:{:.2f}_Normalized:{}_Phase:{}_P:{}_N:{}_loss:{}".format(
                                    dataset_path, distance, int(normal_network), int(phase_inp), plate_number, 60*psf, l
                                )
                                strings.append(folder_name)
    return strings


def get_parameter_strs(
datasets = ['mnist'],
distances = [0.05],
normalized = [False],
phase = [False],
plate_num = [8],
plate_scale_factor = [1],
loss_fn = ['MSE'],
abs_nonlinear = False,
trainable_pixel = [60]):

    strings= []
    for dataset in datasets:
        if dataset == 'mnist':
            dataset_path = 'MNIST'
        elif dataset == 'kmnist':
            dataset_path = 'KMNIST'
        elif dataset == 'cifar10':
            dataset_path = 'CIFAR10'
        print(dataset_path)

        
        for distance in distances:
            for normal_network in normalized:
                if normal_network:
                    network_str = "DiffractionPlateNetworkPoolingNormalized"
                    if abs_nonlinear:
                        network_str = "DiffractionPlateNetworkAbsNonlinearity"
                else:
                    network_str = "DiffractionPlateNetworkPooling"
                    if abs_nonlinear:
                        print('ERROR: Abs Nonlinear Network is always normalized')
                        return -1



                for phase_inp in phase:
                    if phase_inp:
                        phase_str = "--phase_encoding 1"
                    else:
                        phase_str = ""

                    for plate_number in plate_num:
                        for psf in plate_scale_factor:
                            for l in loss_fn:
                                output_p = "/is/cluster/work/lschlieder/DDNN/NormalizingNetwork/{}/D:{:.2f}_Normalized:{}_Phase:{}_P:{}_N:{}_loss:{}_AbsNonlinear:{}".format(
                                    dataset_path, distance, int(normal_network), int(phase_inp), plate_number, 60*psf, l, abs_nonlinear
                                )
                                #print(output_p)
                                param_str = "network_class {} -o {} --pooling_factor 4 --num_layers {} --propagation_size 0.002 --padding 0.0015 --distance {} --plate_scale_factor {} --input_pixel 30 --trainable_pixel {} --epochs 60 --batch_size 32 -M ACC XENT --images_output_steps 1000 --learning_rate 0.1 {} -d {} -E {}\n".format(
                                    network_str, output_p, plate_number, distance, psf, 60, phase_str, dataset, l
                                )
                                #print(param_str)
                                strings.append(param_str)
    return strings


datas = ['mnist', 'kmnist', 'cifar10']
distances = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4]
normalized = [True]
phase = [True]
abs_nonlinear = True
plate_num = [1,2,4,10,16]
plate_scale_factor = [2]
trainable_pixel= [60]
loss_fn = ['MSE','XENT']


for d in datas:
    print(d)
    #print(datas)
    output_strings = get_parameter_strs(plate_num = plate_num, datasets = [d], distances= [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4], phase = phase, normalized = normalized, loss_fn = loss_fn, plate_scale_factor = plate_scale_factor, trainable_pixel = trainable_pixel, abs_nonlinear= abs_nonlinear)
    with open("./output_compare_plates_{}_abs_nonlinear.txt".format(d), 'w') as f:
        for s in output_strings:
            f.write(s)