import numpy as np
import pandas as pd
import os
from bayesnegf import transmission as tran
from bayesnegf import iv as IV
from dynesty.utils import quantile
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
import argparse

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def calculate_single_state(args):
    i, N, t1, t2, eps, alpha, gammaL, gammaR, E_start, E_end, E_points, V_array = args
    
    H = tran.SSH(N, eps, t1, t2)
    E, TE_trace = tran.transmission(N, E_start, E_end, H, gammaL, gammaR, V=0.0, alpha=alpha, E_points=E_points, g=1.0, starkON=True)
    
    I_trace = np.zeros_like(V_array)
    for V_idx, V in enumerate(V_array):
        I_trace[V_idx] = IV.current(N, H, gammaL, gammaR, V, E_start, E_end, alpha=alpha, g=1.0, kT=0.0256, E_points=E_points, starkON=True) / 1e-9
        
    # Return i so loop knows where to store results
    return i, TE_trace, I_trace

def process_results():
    
    parser = argparse.ArgumentParser(description="Post-process Dynesty chains.")
    parser.add_argument("--raw_file", required=True, type=str, help="Path to RawChain.txt")
    parser.add_argument("--save_dir", required=True, type=str, help="Directory to save outputs")
    parser.add_argument("--N", required=True, type=int, help="Number of carbon atoms")
    parser.add_argument("--exp_file", required=True, type=str, help="Path to experimental IV median .txt")
    
    # Slicing arguments (Optional)
    parser.add_argument("--split_param", type=str, default=None, help="Parameter to split by (e.g. eps)")
    parser.add_argument("--split_op", type=str, default=None, help="Operator (<, <=, >, >=)")
    parser.add_argument("--split_val", type=float, default=None, help="Value to split at")
    parser.add_argument("--mode_name", type=str, default="Full_Distribution", help="Label for split")
    
    args = parser.parse_args()

    base_name = os.path.basename(args.raw_file).replace("_RawChain.txt", "")
    df = pd.read_csv(args.raw_file, sep='\t')

    exp_data = np.loadtxt(args.exp_file)
    V_exp = exp_data[:, 0] 
    
    # Apply splitting if arguments are provided
    if args.split_param and args.split_op and args.split_val is not None:
        p, op, v = args.split_param, args.split_op, args.split_val
        if op == '<': df_final = df[df[p] < v]
        elif op == '<=': df_final = df[df[p] <= v]
        elif op == '>': df_final = df[df[p] > v]
        elif op == '>=': df_final = df[df[p] >= v]
        mode_name = args.mode_name
    else:
        df_final = df
        mode_name = "Full_Distribution"

    # Saving the parameter table for plotting or further extraction
    os.makedirs(args.save_dir, exist_ok=True)
    param_save_path = os.path.join(args.save_dir, f"{base_name}_{mode_name}_Parameters.txt")
    df_final.to_csv(param_save_path, index=False, sep='\t')
    print(f"\nCleaned parameter table saved to: {param_save_path}")
    
    weights = df_final['Weight'].values
    weights = weights / np.sum(weights)

    print(f"\nCalculating theoretical envelopes for {len(df_final)} 'unique' states")
    
    # Measurement grid(s)
    E_points = 1000
  
    
    V_array = V_exp
    V_points = len(V_array)
    E_array = np.linspace(-4, 4, E_points)
    E_start, E_end = -4.0, 4.0

    all_TE_curves = np.zeros((len(df_final), E_points))
    all_IV_curves = np.zeros((len(df_final), V_points))

    tasks = []
    for i, row in enumerate(df_final.itertuples()):
        t1 = row.BO * row.t_scale
        t2 = (1 - row.BO) * row.t_scale
        tasks.append((i, args.N, t1, t2, row.eps, row.alpha, row.gammaL, row.gammaR, E_start, E_end, E_points, V_array))

    ncpu = max(1, cpu_count() - 4)

    with Pool(processes=ncpu, maxtasksperchild=10) as pool:
        for i, TE_trace, I_trace in tqdm(pool.imap_unordered(calculate_single_state, tasks), total=len(tasks), desc="Calculating Traces"):
            # Map results to pre-allocated rows
            all_TE_curves[i, :] = TE_trace
            all_IV_curves[i, :] = I_trace

    print("Extracting weighted Medians and 1-Sigma Bounds")

    # Arrays for envelope bounds
    TE_median, TE_lower, TE_upper = np.zeros(E_points), np.zeros(E_points), np.zeros(E_points)
    IV_median, IV_lower, IV_upper = np.zeros(V_points), np.zeros(V_points), np.zeros(V_points)

    # Calculate quantiles ptbypt across the energy/voltage grids
    for j in range(E_points):
        q_TE = quantile(all_TE_curves[:, j], q=[0.16, 0.50, 0.84], weights=weights)
        TE_lower[j], TE_median[j], TE_upper[j] = q_TE

    for j in range(V_points):
        q_IV = quantile(all_IV_curves[:, j], q=[0.16, 0.50, 0.84], weights=weights)
        IV_lower[j], IV_median[j], IV_upper[j] = q_IV

    # Save the single envelope files
    iv_filename = os.path.join(args.save_dir, f"{base_name}_{mode_name}_IV_Envelope.txt")
    te_filename = os.path.join(args.save_dir, f"{base_name}_{mode_name}_TE_Envelope.txt")

    np.savetxt(iv_filename, np.column_stack((V_array, IV_median, IV_lower, IV_upper)), 
               header="Voltage\tMedian_I\tLower_1Sigma\tUpper_1Sigma", comments="", delimiter="\t")
    
    np.savetxt(te_filename, np.column_stack((E_array, TE_median, TE_lower, TE_upper)), 
               header="Energy\tMedian_TE\tLower_1Sigma\tUpper_1Sigma", comments="", delimiter="\t")

    print(f"\nEnvelopes saved to {args.save_dir}")

if __name__ == "__main__":
    process_results()
