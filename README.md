# BayesNEGF

## About

Bayesian inference on single-molecule electronics data for quantum transport. Uses dynamic nested sampling (dynesty) and non-equilibrium Green's functions (NEGF) to parameterise transport models and extract transmission spectra.


### Current Version

Barebones implementation in v0.1.0 similar to the initial scripts found in our recent work.  
Plans to extend this to a more user friendly general framework in the future are underway.
This codebase currently only supports the SSH tight-binding model.


## How to Cite

If you use bayesnegf or its code, please cite according to the CITATION.cff file.


## Installation

Clone the repository and install (requires python >= 3.10):

```bash
git clone https://github.com/Molecular-Electronics-Liverpool/bayesnegf.git
cd bayesnegf
pip install .
```


## Repository Structure & Modules

| Module | Description |
| --- | --- |
| src/bayesnegf/transmission.py | SSH Hamiltonian, Stark shifts, WBL self-energies, NEGF transmission function calculation. |
| src/bayesnegf/iv.py | Landauer–B&uuml;ttiker current integration. |
| src/bayesnegf/optimisation.py | Prior transform and log-likelihood functions for dynesty |
| scripts/run_bayesnegf.py | Runs dynamic nested sampling on exp median I-V file in parallel. |
| scripts/post_process.py | Slices posterior distributions and calculates the weighted I-V and T(E) and credible interval |
| scripts/extract_parameters.py | Calculates the weighted medians, credible intervals and derived quantities of the posterior parameters  |
| scripts/plot_corner.py | Produces the weighted corner plots of the posterior space. |


## Example Usage (Cn = 8 SSH chain)

**1. Run the sampling (args: input_file, output_dir, N_atoms)**  
```bash
python scripts/run_bayesnegf.py ./data/C8_Med.txt ./Processed 8
```
**2. Post-process the SSH chains and generate I-V and T(E) envelopes (run with --help for slicing options)** 
```bash 
python scripts/post_process.py --raw_file ./Processed/Cn_Med_RawChain.txt --save_dir ./Processed --N 8 --exp_file ./data/C8_Med.txt
```


## Future Plans

**Planned extensions:**  

* **The Single Level Model - SLM (Breit-Wigner):** The simplest model a molecular junction can take. With optional temperature dependence.
  
* **The 2-Level Model:** A step up from the SLM, for tracking transport gap trends.
  
* **o/m/p-Benzene Model:** An abstracted model for archetypal quantum interfering systems to locate DQI nodes.
  
* **Joint-Conditioning:** Enabling combinations of I-V curves, dI/dV, d2I/dV2 (from analogue sources e.g. lock-in methods), polar environments etc to improve the likelihood function and increase constraints.
