# PLATO Light Curve Classification Pipeline
![MASS-UBMATF](https://img.shields.io/badge/MASS--UBMATF-Computational_Astrobiology_2026-blue)

## Background

### Introduction
The search for exoplanets has been revolutionized by space-based transit photometry, driven primarily by the successes of the Kepler and TESS missions. While Kepler demonstrated that terrestrial planets are ubiquitous across the galaxy [1], and TESS focuses on short-period planets around bright, nearby stars [3], a critical gap remains: the discovery and precise characterization of Earth-like planets in the habitable zones of bright, Sun-like stars.

### The PLATO Mission
**PLATO** (PLAnetary Transits and Oscillations of stars) is an upcoming European Space Agency (ESA) mission dedicated to discovering and characterizing exoplanets [1]. Its primary objective is to find Earth-like planets orbiting within the habitable zones of Sun-like stars. To achieve this, PLATO will perform ultra-high-precision, long-term photometry to detect planetary transits. Crucially, the mission also focuses on asteroseismology—measuring the tiny acoustic oscillations of the host stars—to determine stellar masses, radii, and ages with unprecedented accuracy. 

### PSLS: Simulating the Universe
The **PLATO Solar-like Light-curve Simulator (PSLS)** is a specialized software tool developed to generate highly realistic synthetic light curves that mimic exactly what the PLATO cameras will see [4]. PSLS physically models stellar granulation, acoustic oscillations, and magnetic activity like starspots and flares. Furthermore, it injects realistic instrumental artifacts, including photon noise, telemetry gaps, and the spacecraft's drift. This provides a rigorous testing ground for data pipelines.

### Project Motivation
Detecting an Earth-sized planet around a Sun-like star requires identifying a brightness drop of barely 0.01%. In raw data, this tiny signal is constantly fighting against noise, aggressive stellar activity (which creates sudden upward spikes) and spacecraft thermal drift (which creates slow, wandering baselines and sudden mask-update jumps). 

Traditional detection algorithms often struggle to disentangle these overlapping astrophysical and instrumental signals. This project tries to solve that problem. By using PSLS to generate a massive, labeled dataset of messy light curves, we can directly evaluate how well modern machine learning handles the noise. Specifically, this repository conducts a head-to-head comparison between Deep Learning architectures (1D Convolutional Neural Networks learning directly from waveforms) and classical Machine Learning (Random Forests relying on expert-engineered features) to determine the most robust strategy for next-generation exoplanet discovery.

This repository contains a Computational Astrobiology project for classifying synthetic PLATO mission light curves. The workflow generates physically motivated stellar light curves with PSLS, injects astrophysical events, corrects instrumental drift, and trains machine-learning models to identify four event classes:

| Class | Label | Meaning |
| --- | --- | --- |
| 0 | neither | Quiet stellar photometry with no injected flare or transit |
| 1 | flare | Stellar flare injected |
| 2 | transit | Planetary transit injected |
| 3 | both | Stellar flare and planetary transit both injected |

## How to Run

Create an environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd ../..
```

Some of the files are large, and need to be installed via LFS:

```bash
git lfs install
git lfs pull
```

Additionally, the stellar model grid 'PlatoLightCurves/psls-1.9/grid_plato.hdf5' designed to capture oscillations of Sun-like stars is too large to be included in this repository. It was provided by directly contacting [mailto:reza.samadi@obspm.fr].

Live run:

```text
PlatoLightCurves/scripts/analyze_classifiers.ipynb
```

For the checks, run:

```bash
pytest --nbmake --nbmake-timeout=60 PlatoLightCurves/scripts/analyze_classifiers.ipynb
nbqa pylint PlatoLightCurves/scripts/analyze_classifiers.ipynb
```


## Repository Structure

The current project layout is:

```text
.
|-- README.md
|-- requirements.txt
|-- PlatoLightCurves/
|   |-- data/
|   |   |-- configs/              # PSLS YAML files for each simulated system
|   |   |-- input_labels/         # Label CSV files used by the notebooks
|   |   |-- models/               # Saved model artifacts and analysis outputs
|   |   `-- outputs/              # Simulated .dat light curve folders
|   |-- psls-1.9/                 # Bundled PSLS simulator and stellar grid
|   `-- scripts/
|       |-- analyze_classifiers.ipynb
|       `-- preparation/
|           |-- data_generation.ipynb
|           |-- preprocessing.ipynb
|           |-- CNN.ipynb
|           |-- random_forest.ipynb
|           `-- run_simulations.py
`-- Tutorials/                    # Tutorial notebooks, not the final project
```

Important data files:

- `PlatoLightCurves/data/input_labels/paired_simulation_labels.csv`: original no-drift label table.
- `PlatoLightCurves/data/input_labels/paired_simulation_labels_drift.csv`: drift-injected label table.
- `PlatoLightCurves/data/input_labels/paired_simulation_labels_combined.csv`: final table combining both samples above.
- `PlatoLightCurves/data/outputs/sys_XXXX/*.dat`: simulated PSLS light curves.
- `PlatoLightCurves/data/outputs/sys_XXXX/*_driftcorrected.dat`: drift-corrected light curves produced by preprocessing.
- `PlatoLightCurves/data/models/rf_checkpoint.joblib`: saved random forest pipeline.
- `PlatoLightCurves/data/models/rf_full_analysis.csv`: validation-set predictions merged with physical labels.
- `PlatoLightCurves/data/models/cnn_dataset.pt`, `cnn_model.pt`, and `split_indices.pt`: CNN cache/checkpoint artifacts tracked through Git LFS.

The combined data contains 1,490 simulated systems. 1,000 are without drift, and among those: 500 quiet, 500 flare-only, 500 transit-only, and 500 flare-plus-transit examples. The remaining 490 samples have drifts distributed across `none`, `low`, `medium`, `high`, and `max`, with the same four-class distribution as described above.

## Notebook Guide

### `PlatoLightCurves/scripts/data_generation.ipynb`

This notebook defines the labeled synthetic dataset. It loads the PSLS base configuration and stellar grid, samples valid stellar parameters, assigns each system one of the four anomaly classes, and prepares per-system YAML files under `PlatoLightCurves/data/configs/`.

The notebook controls the injected astrophysical signals:

- For flare-positive systems, it enables the PSLS flare model and samples flare amplitude, duration, and timing parameters.
- For transit-positive systems, it enables the transit model and samples planet radius and orbital period.
- For mixed systems, it enables both flare and transit components.
- For quiet systems, both event injectors remain disabled while stellar and instrumental variability are still present.

The first generated batch represents no-drift systems. A second batch introduces drift categories so that the classifiers can be evaluated under harder instrumental conditions. The notebook then combines the clean and drift label tables into `PlatoLightCurves/data/input_labels/paired_simulation_labels_combined.csv`.

It also contains plots and summary checks for the sampled parameter distributions, including stellar mass, effective temperature, surface gravity, flare amplitude, transit radius, and orbital period.

### `PlatoLightCurves/scripts/preprocessing/run_simulations.py`

This script batch-runs PSLS for the generated YAML configs. For each system ID, it finds the matching config, creates an output directory, and runs `psls.py`.

### `PlatoLightCurves/scripts/preprocessing/preprocessing.ipynb`

This notebook corrects instrumental drift artifacts in the simulated light curves. Drift matters because slow baseline changes and sudden mask-update jumps can make a transit look shallower, hide a flare, or create false event-like structure.

The notebook performs two main corrections:

- Jump correction: it identifies mask-update locations from the third column of each `.dat` file, compares median flux before and after each jump, and shifts later segments so the baseline remains continuous.
- Slope correction: it estimates a low-frequency baseline with a downsampled Savitzky-Golay filter, interpolates that baseline back to the full cadence, and subtracts it from the light curve.

The notebook includes diagnostic functions that report segment medians, residual slopes, original slopes, and variance reduction. It writes corrected files named `*_driftcorrected.dat` next to the original `.dat` files in `PlatoLightCurves/data/outputs/sys_XXXX/`.

### `PlatoLightCurves/scripts/CNN.ipynb`

This notebook trains a one-dimensional convolutional neural network directly on light-curve waveforms. It treats the problem as two binary questions rather than one four-class question: does the system contain a flare, and does it contain a transit? The two outputs can represent all four event classes.

The first stage is dataset caching. The notebook loads raw or drift-corrected light curves, interpolates each system to a fixed 50,000-point grid, and saves the result as `PlatoLightCurves/data/models/cnn_dataset.pt`. It also stores labels, system IDs, flare amplitudes, and planet radii so that later analysis can connect model performance to physical parameters.

The model architecture is `LightCurveCNN`. It uses four convolution blocks with batch normalization, ReLU activations, and pooling. The kernels are designed to capture local time-series shapes: sharp flare spikes, transit ingress and egress, and broader waveform patterns. After adaptive pooling, a shared fully connected network feeds two separate heads, one for flares and one for transits.

Training uses Adam, binary cross-entropy style losses, and validation metrics for each head. The notebook reports precision, recall, F1 score, and confusion statistics. It also includes analysis of detection probability versus flare amplitude and transit radius, which is important because small planets and weak flares are the hardest cases.

### `PlatoLightCurves/scripts/random_forest.ipynb`

This notebook trains a random forest classifier. It reads the same cached light curves used by the CNN from `PlatoLightCurves/data/models/cnn_dataset.pt`, extracts hand-crafted features, applies the same saved train/validation split from `split_indices.pt`, fits a scikit-learn random forest, and saves the trained pipeline as `PlatoLightCurves/data/models/rf_checkpoint.joblib`.

The feature extraction step converts each 50,000-point waveform into a compact feature vector. The features include:

- Shape statistics such as skewness, kurtosis, negative-outlier fraction.
- Flare indicators such as high-percentile signal-to-noise, positive flux area, fraction of points above 3-sigma, and longest high-flux run.
- Transit indicators from Box Least Squares, including best depth, depth signal-to-noise, and peak BLS power.

After training, the notebook prints a confusion matrix and classification report. It then builds `PlatoLightCurves/data/models/rf_full_analysis.csv`, which merges validation predictions with the physical metadata from `paired_simulation_labels_combined.csv`. 

### `PlatoLightCurves/scripts/analyze_classifiers.ipynb`

This notebook is a short follow-up analysis workspace intended for comparing saved classifiers and loading model artifacts. It currently imports the scientific stack and begins from saved `.pt` or `.joblib` outputs rather than rebuilding simulations.

## End-to-End Workflow

The project flow is:

```text
PSLS stellar grid and base YAML
        |
        v
data_generation.ipynb
        |---> data/configs/sim_XXXX.yaml
        |---> data/input_labels/*.csv
        |
        v
run_simulations.py
        |---> data/outputs/sys_XXXX/*.dat
        |
        v
preprocessing.ipynb
        |---> data/outputs/sys_XXXX/*_driftcorrected.dat
        |
        +------------------------------+
        |                              |
        v                              v
CNN.ipynb                      random_forest.ipynb
        |                              |
        v                              v
data/models/*.pt               data/models/rf_checkpoint.joblib
                               data/models/rf_full_analysis.csv
```

## References

[1] Borucki, W. J., Koch, D., Basri, G., Batalha, N., Brown, T., Caldwell, D., ... & Prsa, A. (2010). Kepler planet-detection mission: introduction and first results. Science, 327(5968), 977-980.

[2] Rauer, H., Catala, C., Aerts, C., Appourchaux, T., Benz, W., Brandeker, A., ... & Tkachenko, A. (2014). The PLATO 2.0 mission. Experimental Astronomy, 38(1), 249-330.

[3] Ricker, G. R., Winn, J. N., Vanderspek, R., Latham, D. W., Bakos, G. Á., Bean, J. L., ... & Villasenor, J. (2015). Transiting exoplanet survey satellite. Journal of Astronomical Telescopes, Instruments, and Systems, 1(1), 014003-014003.

[4] Samadi, R., Deru, A., Reese, D., Marchiori, V., Grolleau, E., Green, J. J., ... & Smith, A. M. S. (2019). The PLATO Solar-like Light-curve Simulator-A tool to generate realistic stellar light-curves with instrumental effects representative of the PLATO mission. Astronomy & Astrophysics, 624, A117.

