import os 
import numpy as np



def get_arguments(layers, output_size, datasets, offsets, intensity_output, output_folder = 'work/lschlieder/OpticalRegression/DDNN/size_sweep/'):
    output_args = []
    for l in layers:
        for os in output_size:
            for d in datasets:
                for of in offsets:
                    for io in intensity_output: 
                        print(io, ' ', of)
                        
                        output_file_str = f"offset:{of}/intensity_output:{io}/{d}/layers:{l}_output-size:{os}_{d}_offset:{of}_intensity_output:{io}"  
                        ofstr = "--offset 1"if of else ""
                        iostr = "--intensity_output 1" if io else ""
                        lr = "--learning_rate 0.0001" if os < 4 else ""
                        arguments = f"network_class OpticalFunctionApproximatorPhase -o {output_folder}{output_file_str} -d {d} -e 160 -L 0.003584 -P 224 -N 224 --encoding_pixels 112 --num_layers {l} --mean_size {os} {ofstr} {iostr} {lr}\n"
                        output_args.append(arguments)
    return output_args

def get_file_names(layers, output_size, datasets, offsets, intensity_output):
    output_files= []
    for l in layers:
        for os in output_size:
            for d in datasets:
                for of in offsets:
                    for io in intensity_output:
                        output_file_str = f"offset:{of}/intensity_output:{io}/{d}/layers:{l}_output-size:{os}_{d}_offset:{of}_intensity_output:{io}"  
                        #ofstr = "--offset 1"if of else ""
                        #arguments = f"network_class OpticalFunctionApproximatorPhase -o {output_folder}{output_file_str} -d {d} -e 160 --num_layers {l} --mean_size {os} {ofstr} \n"
                        output_files.append(output_file_str)
    return output_files

#arguments_initial_training_data = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/Regression/initial_training_data_collection.txt"
#arguments_bigger_phase_plates = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/Regression/comparison_runs_bigger_plates.txt"
arguments_all = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/Regression/arguments_gridsearch_all.txt"

arguments_1 = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/Regression/arguments_gridsearch_batch_1.txt"
arguments_2 = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/Regression/arguments_gridsearch_batch_2.txt"
arguments_3 = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/Regression/arguments_gridsearch_batch_3.txt"
arguments_4 = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/Regression/arguments_gridsearch_batch_4.txt"

arguments = [arguments_all, arguments_1, arguments_2, arguments_3, arguments_4]
#arguments_comparison = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/Regression/comparison_runs_intensity_output.txt"
layers = [1,3,6,9,15,20]
#layers = [3]
output_size  = [2,4,6,16,32,64,112]
#datasets = ['regression_const', 'regression_linear', 'regression_neg_linear', 'regression_sin_big']
datasets = ['regression_sin', 'regression_x2', 'regression_steps', 'regression_sin2x', 
               'regression_sin3x', 'regression_sin4x', 'regression_x2exp', 'regression_linear',
               'regression_neg_linear', 'regression_const', 'regression_sin_big', 'regression_sin_x2']
datasets_1 = ['regression_sin', 'regression_x2', 'regression_steps', 'regression_sin2x']
datasets_2 = ['regression_sin3x', 'regression_sin4x', 'regression_x2exp', 'regression_linear']
datasets_3 = ['regression_neg_linear', 'regression_const', 'regression_sin_big', 'regression_sin_x2']
datasets_4 = ['regression_sin2x']
#datasets_size_sweep = ['regression_sin2x', 'regression_x2exp', 'regression_sin_big']
offset =[True, False]
intensity_output = [True]

args = get_arguments(layers, output_size, datasets, offset, intensity_output, '/is/cluster/work/lschlieder/DDNN/Regression/gridsearch/')
args1 = get_arguments(layers, output_size, datasets_1, offset, intensity_output, '/is/cluster/work/lschlieder/DDNN/Regression/gridsearch/')
args2 = get_arguments(layers, output_size, datasets_2, offset, intensity_output, '/is/cluster/work/lschlieder/DDNN/Regression/gridsearch/')
args3 = get_arguments(layers, output_size, datasets_3, offset, intensity_output, '/is/cluster/work/lschlieder/DDNN/Regression/gridsearch/')
args4 = get_arguments(layers, output_size, datasets_4, offset, intensity_output, '/is/cluster/work/lschlieder/DDNN/Regression/gridsearch/')

args_arr = [args, args1, args2, args3, args4]
#args_size_sweep = get_arguments([3], output_size, datasets_size_sweep, offset,intensity_output, '/is/cluster/work/lschlieder/DDNN/Regression/size_sweep_intensity/')

for i,a in enumerate(arguments):
    with open(a, 'w') as f:
        for s in args_arr[i]:
            f.write(s)

##with open(arguments_size_sweep, 'w') as f:
#    for s in args_size_sweep:
#        f.write(s)
