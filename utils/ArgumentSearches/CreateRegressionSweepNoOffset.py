import os 
import numpy as np
from tqdm import tqdm

layers = [1,3,4,10,20]
encodings = ['IntensityPhase', 'Intensity','Phase', 'RealSetup']
#datasets = ['regression_sin','regression_x2', 'regression_steps', 'regression_sin2x', 'regression_sin3x', 'regression_sin4x',
#            'regression_sin_x2', 'regression_x2exp','regression_linear', 'regression_neg_linear',
#            'regression_const', 'regression_sin_big', 'regression_big_sin2x', 'regression_big_sin3x', 'regression_big_sin4x',
#           'regression_neg_sin',]


#datasets = ['regression_small_step']

#datasets = ['regression_sin4x','regression_sin2x']
datasets = ['regression_x2exp', 'regression_small_step', 'regression_relu', 'regression_sin3x', 'regression_sin2x', 'regression_neg_sin']
#neg_output = ['--negative_output True ', ""]

neg_output = ['--negative_output True ']
#neg_output = [""]

#radius = np.linspace(0.0001,0.0003, 3)
#radius = np.linspace(0.0001,0.0003, 3)
#radius = np.array([0.00002, 0.00004, 0.00006, 0.0004])
radius = [8.833759103265864e-05, 0.00011625880792490716, 0.00012492821930575377, 0.00015708891926581586, 0.00018681126047899625]
#radius = [0.00003]
#offset = ['']
N = 112
L = 0.001792

output = '/is/cluster/work/lschlieder/DDNN/UniversalFunctionApproximator/Regression/NoOffset/'

output_strs = [[f'''network_class Regression.OpticalFunctionApproximatorPhase -o {output}{i_dataset}/encoding:{i_encoding}/negative_output_weights:{i_neg_output =='--negative_output True '}
/radius:{i_radius:.6f}/Layers:{i_layers}_offset:False 
-d {i_dataset} --num_layers {i_layers} {i_neg_output}--radius {i_radius} --output_distance 0.0008 --output_function sum --learning_rate 0.01 
--callbacks TB TB-Regression ModelCheckpoint BackupAndRestore LearningRateSchedule --encoding {i_encoding}'''.replace('\n', '')
                for i_neg_output in neg_output
                for i_radius in radius
                for i_layers in layers
                for i_encoding in encodings] for i_dataset in datasets]

for s in output_strs:
    if "  " in s:
        print(repr(s))
    assert("  " not in s)

print("Number of runs: ",len(output_strs[0]))
print(output_strs[0])

argument_files = [f"/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/Regression/RegressionRunsNoOffset_neg_output_{i_dataset}.txt" for i_dataset in datasets]
#argument_file_mnist = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/Classification/ClassificationPaperRuns_mnist.txt"
#argument_file_kmnist = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/Classification/ClassificationPaperRuns_kmnist.txt"
#argument_file_cifar10 = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/Classification/ClassificationPaperRuns_cifar10.txt"
#argument_file = "/home/lennart/Desktop/TestFile.txt"

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

