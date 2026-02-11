import os 
import numpy as np
from tqdm import tqdm

layers = []
num_outputs = [ 1, 3, 6, 10, 15, 20, 30, 40, 50, 70, 90, 100, 200, 1000, 2000, 10000]
encodings = ['phase', 'amp', 'both']

datasets = ['regression_sin','regression_x2', 'regression_steps', 'regression_sin2x', 'regression_sin3x', 'regression_sin4x',
            'regression_sin_x2', 'regression_x2exp','regression_linear', 'regression_neg_linear',
            'regression_const', 'regression_sin_big', 'regression_big_sin2x', 'regression_big_sin3x', 'regression_big_sin4x',
            'regression_neg_sin']


#datasets = ['regression_small_step']

#datasets = ['regression_sin4x','regression_sin2x']
#datasets = ['regression_x2exp', 'regression_small_step', 'regression_relu', 'regression_sin3x', 'regression_sin2x', 'regression_neg_sin']
#neg_output = ['--negative_output True ', ""]

#neg_output = ['--negative_output True ']

#N = 112
#L = 0.001792

output = '/is/cluster/work/lschlieder/DDNN/UniversalFunctionApproximator/Regression/NoPropagation/'

output_strs = [[f'''network_class Regression.RegressionModelNoPropagation -o {output}{i_dataset}/encoding_{i_encoding}
/num_outputs_{i_num} -d {i_dataset} --num_outputs {i_num} --learning_rate 0.001 
--callbacks TB TB-Regression ModelCheckpoint BackupAndRestore LearningRateSchedule --input_enc {i_encoding} --mode negative'''.replace('\n', '')
                for i_num in num_outputs
                for i_encoding in encodings] for i_dataset in datasets]

for s in output_strs:
    if "  " in s:
        print(repr(s))
    assert("  " not in s)

print("Number of runs: ",len(output_strs[0]))
print(output_strs[0])

argument_files = [f"/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/Regression/NoPropagation/{i_dataset}.txt" for i_dataset in datasets]
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