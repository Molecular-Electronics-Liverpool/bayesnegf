# BayesNEGF

## About

Bayesian inference on single-molecule electronics data for quantum transport. Uses dynamic nested sampling (dynesty) and non-equilibrium Green's functions (NEGF) to parameterise transport models and extract transmission spectra.

### Current Version

Barebones implementation in v0.1.0 similar to the initial scripts found in our recent work.  
Plans to extend this to a more user friendly general framework in the future are underway.
This codebase currently only supports the SSH tight-binding model.

### Future Plans

I plan to extend the framework with the following features:
* **The Single Level Model - SLM (Breit-Wigner):** The simplest model a molecular junction can take. With optional temperature dependence.
* **The 2-Level Model:** A step up from the SLM, for tracking transport gap trends.
* **o/m/p-Benzene Model:** An abstracted model for archetypal quantum interfering systems to local DQI nodes.
* **Joint-Conditioning:** Enabling combinations of I-V curves, dI/dV, d2I/dV2 (from analogue sources e.g. lock-in methods), polar environments etc to improve the likelihood function and increase constraints.
* **Median Bootstrapping:** Better estimation of the median error which would be smaller than the current implementation ensemble deviations. Improving and shrinking the credible interval bands.
