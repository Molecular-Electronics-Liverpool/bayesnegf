import pandas as pd
import numpy as np
import corner
import matplotlib.pyplot as plt
import os

raw_file = input("Enter the full path to your RawChain.txt file: ").strip('\"\'')
save_dir = os.path.dirname(raw_file)
df = pd.read_csv(raw_file, sep='\t')

# Normalise weights
weights = df['Weight'].values
weights = weights / np.sum(weights)

labels = ["BO", "t_scale", "eps", "alpha", "gammaL", "gammaR"]
data = df[labels].values

q16, median_BO, q84 = corner.quantile(df['BO'].values, [0.165, 0.5, 0.835], weights=weights)
upper_err = q84 - median_BO
lower_err = median_BO - q16

print(f"Bond Order = {median_BO:.3f} +{upper_err:.3f} / -{lower_err:.3f}")

print("Weighted corner plot")

fig = corner.corner(
    data, 
    weights=weights, 
    labels=labels,
    quantiles=[0.165, 0.5, 0.835], # 67% credible interval - +/- 1sigma in approximate gaussian terms
    show_titles=True,
    smooth=1.0,
    smooth1d=1.0,
    bins=20,
    hist_bin_factor=1.0
)

plot_path = os.path.join(save_dir, "Cn_Weighted_Diagnostic_Corner.png")
fig.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved to: {plot_path}")
plt.show()