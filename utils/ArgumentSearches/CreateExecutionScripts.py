import os 
import numpy as np
from tqdm import tqdm



datasets = ['regression_sin','regression_x2', 'regression_steps', 'regression_sin2x', 'regression_sin3x', 'regression_sin4x',
            'regression_sin_x2', 'regression_x2exp','regression_linear', 'regression_neg_linear',
            'regression_const', 'regression_sin_big', 'regression_big_sin2x', 'regression_big_sin3x', 'regression_big_sin4x',
            'regression_neg_sin']

for d in datasets: 

    output_str = f'''executable = /lustre/home/lschlieder/execution_script_DDNN.sh
    arguments = $(args)
    error = {d}.err
    output = {d}.out
    log = {d}.log
    request_memory = 100000
    request_gpus = 1
    requirements = TARGET.CUDACapability > 7.5
    requirements = TARGET.CUDAGlobalMemoryMb > 10000
    +MaxRunningPrice = 2000
    +RunningPriceExceededAction = "restart"
    queue args from ../Arguments/Optical/Regression/NoPropagation/{d}.txt
    '''

    s = f"/home/lennart/Desktop/CLUSTER_HOME_NET/ExecutionScriptsRegression/execution_script_{d}"
    with open(s, 'w') as f:
        f.write(output_str)