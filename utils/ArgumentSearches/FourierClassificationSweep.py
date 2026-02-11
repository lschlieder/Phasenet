import os 
import numpy as np

layers = [3]
encodings = ['TrigPolynomialPhase']
datasets = ['mnist_cat', 'kmnist_cat', 'cifar10_cat']
neg_output = ['--negative_output True ', ""]
#num_hyper_layers = [2,3]
fourier_coefficients = [2,3,4]
radius = [0.0001, 0.0002, 0.0003]
N = 112

L = 0.001792

output = '/is/cluster/work/lschlieder/DDNN/UniversalFunctionApproximator/Classification/FourierEnc/'



output_strs = [[
    f'''network_class OutputSizeModel -o {output}{i_dataset}/encoding:{i_encoding}/negative_output_weights:{i_neg_output =='--negative_output True '}/layers_{i_layers}/radius_{i_rad}/fourier_components_{i_fcoeff} -L {L} -N {N} -s 1 -d {i_dataset} 
-M CATACC -E CATXENT --callbacks TB ModelCheckpoint BackupAndRestore --images_output_steps 30000 --learning_rate 0.001 {i_neg_output}
--num_layers {i_layers} --intensity_output True --last_propagation 0.04 --output_shape rect --radius {i_rad} --num_coefficients {i_fcoeff} 
--data_size {N//i_fcoeff} 
--propagation_distance 0.08 --encoding {i_encoding}'''.replace('\n', '')
    for i_neg_output in neg_output
    for i_layers in layers
    for i_encoding in encodings
    for i_rad in radius
    for i_fcoeff in fourier_coefficients] for i_dataset in datasets
]

for s in output_strs:
    if "  " in s:
        print(repr(s))
    assert("  " not in s)

print("Number of runs: ",len(output_strs[0]))


argument_file_mnist = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/Classification/ClassificationPaperRunsFourier_mnist.txt"
argument_file_kmnist = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/Classification/ClassificationPaperRunsFourier_kmnist.txt"
argument_file_cifar10 = "/home/lennart/Desktop/CLUSTER_HOME_NET/Arguments/Optical/Classification/ClassificationPaperRunsFourier_cifar10.txt"
#argument_file = "/home/lennart/Desktop/TestFile.txt"

argument_files = [argument_file_mnist, argument_file_kmnist,argument_file_cifar10]

for j,s in enumerate(datasets):
    with open(argument_files[j], 'w') as f:
        for i,args in enumerate(output_strs[j]):
            f.write(args + '\n')

