import os 
import numpy as np
from tqdm import tqdm

layers = [3]
fourier_components = [ 1,2,4,8,10,20,50,100]
encodings = ['Fourier']


datasets = ['regression_gaussian', 'regression_step', 'regression_big_gaussian', 'regression_big_sin4x', 'regression_big_sin3x', 'regression_big_sin2x', 'regression_neg_sin', 'regression_relu']

neg_output = ['--negative_output True ', '']

#radius = [8.833759103265864e-05, 0.00011625880792490716, 0.00012492821930575377, 0.00015708891926581586, 0.00018681126047899625]
radius = [0.00013]
#radius = [0.00003]
#offset = ['']

N = 112
L = 0.001792


output = '/is/cluster/work/lschlieder/DDNN/UniversalFunctionApproximator/Regression/Fourier/Sigma_0.0009/bigger_functions/'

output_strs = [[f'''network_class Regression.OpticalFunctionApproximatorPhase -o {output}{i_dataset}/regularized/fourier_components_{comp}
/radius:{i_radius:.6f}/Layers:{i_layers}/neg_output:{i_neg_out == '--negative_output True '}/ --last_propagation 0.04 --output_category_shape rect 
-d {i_dataset} --num_layers {i_layers} {i_neg_out}--radius {i_radius} --output_distance 0.0008 --output_function mean --learning_rate 0.01 
--callbacks TB TB-Regression ModelCheckpoint BackupAndRestore LearningRateSchedule --use_regularization True --encoding {i_encoding} --fourier_components {comp} --sigma 0.0009 --offset True'''.replace('\n', '')
                for i_radius in radius
                for i_layers in layers
                for i_encoding in encodings
                for i_neg_out in neg_output
                for comp in fourier_components] for i_dataset in datasets]

for s in output_strs:
    if "  " in s:
        print(repr(s))
    assert("  " not in s)

print("Number of runs: ",len(output_strs[0]))
print(output_strs[0])

argument_files = [f"/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/Regression/FourierEnc/mean/bigger_functions/regularized_{i_dataset}.txt" for i_dataset in datasets]


#argument_files = [argument_file_mnist, argument_file_kmnist,argument_file_cifar10]
print('starting to write files')
for j,s in tqdm(enumerate(argument_files)):
    with open(s, 'w') as f:
        for i, args in enumerate(output_strs[j]):
            f.write(args+ '\n')
#for j,s in enumerate(datasets):
#    with open(argument_files[j], 'w') as f:
#        for i,args in enumerate(output_strs[j]):
 #           f.write(args + '\n')

