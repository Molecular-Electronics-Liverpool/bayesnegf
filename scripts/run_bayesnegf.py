import pandas as pd
import numpy as np
from bayesnegf import optimisation as opt
import time
import os
from multiprocessing import Pool
from multiprocessing import cpu_count
from scipy.interpolate import interp1d
import dynesty
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def downsample_data(V_raw, I_raw, Sigma_raw, target_points=200):
   
    V_unique, unique_indices = np.unique(V_raw, return_index=True)
    I_unique = I_raw[unique_indices]
    Sigma_unique = Sigma_raw[unique_indices]
    
    V_efficient = np.linspace(np.min(V_unique), np.max(V_unique), target_points)

    f_I = interp1d(V_unique, I_unique, kind='linear', fill_value='extrapolate')
    f_Sigma = interp1d(V_unique, Sigma_unique, kind='linear', fill_value='extrapolate')
    
    I_efficient = f_I(V_efficient)
    Sigma_efficient = f_Sigma(V_efficient)
    
    return V_efficient, I_efficient, Sigma_efficient

def run_sampler():
    print(f"Running in parallel with {ncpu} CPUs")
    start_time = time.time()

    ndim = 6 # No of parameters in model

    read_dir = sys.argv[1].strip('\"\'')
    save_dir = sys.argv[2].strip('\"\'')
    N = int(sys.argv[3])
    base_name = os.path.splitext(os.path.basename(read_dir))[0]
    os.makedirs(save_dir, exist_ok=True)
    print(f"Processing on filename: {base_name}")

    # Upload data for IV curves
    print("Loading IV Data")
    df = pd.read_csv(read_dir, header=None, sep='\t', names=['Voltage', 'Current', 'Sigma_Log'])
    V_exp_raw = df['Voltage'].values # Linear scale
    I_exp_raw = df['Current'].values # Linear scale
    log_sigma_exp_raw = df['Sigma_Log'].values / 1.349 # IQR / 1.349 = sigma of the log of the I data

    V_exp, I_exp, log_sigma_exp = downsample_data(V_exp_raw, I_exp_raw, log_sigma_exp_raw, target_points=80)
    print(f"Original Data Size: {len(V_exp_raw)} was downsampled to: {len(V_exp)}")

    dV = (max(V_exp) - min(V_exp)) / len(V_exp)
    print(f"New dV = {dV} V")

    print("Dynesty Dynamic Nested Sampler Start")
    with Pool(processes=ncpu) as pool:

        sampler = dynesty.DynamicNestedSampler(
            opt.log_likelihood, 
            opt.prior_transform, 
            ndim,
            pool=pool,
            queue_size=ncpu,
            sample='rslice',
            slices=5,
            ptform_args=(N,),
            logl_args=(V_exp, I_exp, log_sigma_exp, N, -4.0, 4.0)
        )

        sampler.run_nested(
            nlive_init=1000,        # baseline std
            nlive_batch=200,       # Number of points to add in each posterior batch
            maxbatch=10,
            wt_kwargs={'pfrac': 0.8}  # Switch to 80% parameter estimations while still searching for new minima
        )

        print(f"runtime: {time.time() - start_time:.2f} seconds using {ncpu} CPUs")
    
    # Extracting results and convert to unweighted MCMC format
    results = sampler.results
    weights = np.exp(results['logwt'] - results['logz'][-1])
    samples = results.samples

    # 2D ASCII table layout
    records = []
    for step, (row, weight) in enumerate(zip(samples, weights)):
        records.append([step, weight] + list(row))
    
    param_labels = ["Step", "Weight", "BO", "t_scale", "eps", "alpha", "gammaL", "gammaR"]
    df_raw = pd.DataFrame(records, columns=param_labels)

    save_path = os.path.join(save_dir, f"{base_name}_RawChain.txt")
    df_raw.to_csv(save_path, index=False, sep='\t')

total_cpu = cpu_count()
ncpu = max(1, total_cpu - 2)

if __name__ == "__main__":
    run_sampler()
