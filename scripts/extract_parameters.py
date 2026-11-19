import pandas as pd
import numpy as np
import corner
from bayesnegf import transmission as tran

raw_file = input("Enter the full path to your RawChain.txt file: ").strip('\"\'')
df = pd.read_csv(raw_file, sep='\t')

weights = df['Weight'].values
sum_weights = np.sum(weights)
weights = weights / np.sum(weights)

df['t_triple'] = df['BO'] * df['t_scale']
df['t_single'] = (1 - df['BO']) * df['t_scale']
gammaL_abs = np.abs(df['gammaL'])
gammaR_abs = np.abs(df['gammaR'])

# The ratio of the largest coupling over smallest coupling
g_max = np.maximum(gammaL_abs, gammaR_abs)
g_min = np.minimum(gammaL_abs, gammaR_abs)
df['gammaRatio'] = g_max / g_min
df['gammaAVG'] = (gammaL_abs + gammaR_abs) / 2.0
df['gammaMax'] = g_max

df['sym_factor'] = 4 * gammaL_abs * gammaR_abs / (gammaL_abs + gammaR_abs)**2
df['gamma_eff'] = gammaL_abs * gammaR_abs / (gammaL_abs + gammaR_abs)

N = int(input("Enter number of atoms N: "))

homo_idx = (N // 2) - 1
lumo_idx = N // 2

HLgaps = []
for eps, t_triple, t_single in zip(df['eps'].values, df['t_triple'].values, df['t_single'].values):
    H = tran.SSH(N, eps, t_triple, t_single)
    evals = np.linalg.eigvalsh(H)
    HLgaps.append(evals[lumo_idx] - evals[homo_idx])

df['HLgap'] = HLgaps

labels = ["BO", "t_triple", "t_single", "t_scale", "eps", "alpha", "gammaL", "gammaR", "HLgap", "gamma_eff", "sym_factor", "gammaAVG"]
active_labels = [l for l in labels if l in df.columns]

header_names = []
for param in active_labels:
    header_names.extend([param, f"{param}_upper_err", f"{param}_lower_err"])

row_data = []

for param in active_labels:
    data = df[param].values

    q16, median, q84 = corner.quantile(data, [0.16, 0.5, 0.84], weights=weights)
    upper_err = q84 - median
    lower_err = median - q16

    row_data.extend([f"{median:.3f}", f"{upper_err:.3f}", f"{lower_err:.3f}"])

    w_mean = np.sum(data * weights)
    w_var = np.sum(weights * (data - w_mean)**2)
    w_SD = np.sqrt(w_var)

    print(f"{param:.<10} = {median:.3f} +{upper_err:.3f} / -{lower_err:.3f} (mean: {w_mean:.3f} SD: {w_SD:.3f})")

print("\n--------------------------------------------------------------------------------\n")
print("\t".join(header_names))
print("\t".join(row_data))
print(f"\n posterior probability: {sum_weights}")