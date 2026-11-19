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

## Future Plans

**Planned extensions:**  

* **The Single Level Model - SLM (Breit-Wigner):** The simplest model a molecular junction can take. With optional temperature dependence.
  
* **The 2-Level Model:** A step up from the SLM, for tracking transport gap trends.
  
* **o/m/p-Benzene Model:** An abstracted model for archetypal quantum interfering systems to locate DQI nodes.
  
* **Joint-Conditioning:** Enabling combinations of I-V curves, dI/dV, d2I/dV2 (from analogue sources e.g. lock-in methods), polar environments etc to improve the likelihood function and increase constraints.
