import os 
import numpy as np
from tqdm import tqdm
datasets = ['regression_gaussian', 'regression_step', 'regression_big_gaussian', 'regression_sin4x', 'regression_x2exp', 'regression_neg_sin', 'regression_relu']
output_strs = [f'''executable = /lustre/home/lschlieder/execution_script_DDNN.sh
arguments = $(args)
error = {dataset}_correct_sigma.err
output = {dataset}_correct_sigma.out
log = {dataset}_correct_sigma.log
request_memory = 100000
request_gpus = 1
requirements = TARGET.CUDACapability > 7.5
requirements = TARGET.CUDAGlobalMemoryMb > 10000
+MaxRunningPrice = 2000
+RunningPriceExceededAction = "restart"
queue args from /home/lschlieder/Arguments/Optical/Regression/FourierEnc/Regression_runs_fourier_correct_sigma_{dataset}.txt''' for dataset in datasets]

output_files = [f"/home/lennart/Desktop/CLUSTER_HOME_NET/ExecutionScriptsRegression/fourier/execution_script_correct_sigma_{i_dataset}.txt" for i_dataset in datasets]

for i,s in enumerate(output_files):
    with open(s, 'w') as f:
        f.write(output_strs[i])

    
#dataset_list = ["(\"{d}\" "]


dataset_list = "(" + " ".join(f'"{item}"' for item in datasets) + ")"
print(dataset_list)

run_all_script = f'''#!/bin/bash
# List of dataset names
dataset_names={dataset_list}

delay=0
# Loop through each dataset name
for dataset_name in "${{dataset_names[@]}}"; do
    # Schedule the command to run 30 minutes later using at
    echo $dataset_name
    echo "condor_submit_bid 30 execution_script_correct_sigma_${{dataset_name}}.txt" | at now + $delay minutes

    ((delay+=30))
done
'''

all_script_path = "/home/lennart/Desktop/CLUSTER_HOME_NET/ExecutionScriptsRegression/fourier/execution_script_all.sh"
with open(all_script_path, 'w') as f:
    f.write(run_all_script)

os.chmod(all_script_path, 0o755)