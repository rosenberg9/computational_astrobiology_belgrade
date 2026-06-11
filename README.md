# PLATO Light Curve Classification Pipeline
![MASS-UBMATF](https://img.shields.io/badge/MASS--UBMATF-Computational_Astrobiology_2026-blue)

This repository contains a Computational Astrobiology project for classifying synthetic PLATO mission light curves. The workflow generates physically motivated stellar light curves with PSLS, injects astrophysical events, corrects instrumental drift, and trains machine-learning models to identify four event classes:

| Class | Label | Meaning |
| --- | --- | --- |
| 0 | neither | Quiet stellar photometry with no injected flare or transit |
| 1 | flare | Stellar flare injected |
| 2 | transit | Planetary transit injected |
| 3 | both | Stellar flare and planetary transit both injected |

A light curve records how a star's brightness changes over time. A transiting planet makes a small repeated dip; a stellar flare makes a short brightening. Realistic telescope data also include noise, data gaps, and instrumental drift. This project tests whether models can separate those astrophysical signals from the observing artifacts.

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
|       |-- data_generation.ipynb
|       |-- preprocessing.ipynb
|       |-- CNN.ipynb
|       |-- random_forest.ipynb
|       |-- analyze_classifiers.ipynb
|       `-- run_simulations.py
`-- Tutorials/                    # Course tutorial notebooks, not the final project
```

Important data files:

- `PlatoLightCurves/data/input_labels/paired_simulation_labels.csv`: original no-drift label table.
- `PlatoLightCurves/data/input_labels/paired_simulation_labels_drift.csv`: drift-injected label table.
- `PlatoLightCurves/data/input_labels/paired_simulation_labels_combined.csv`: final 2,000-row label table used for modeling.
- `PlatoLightCurves/data/outputs/sys_XXXX/*.dat`: simulated PSLS light curves.
- `PlatoLightCurves/data/outputs/sys_XXXX/*_driftcorrected.dat`: drift-corrected light curves produced by preprocessing.
- `PlatoLightCurves/data/models/rf_checkpoint.joblib`: saved random forest pipeline.
- `PlatoLightCurves/data/models/rf_full_analysis.csv`: validation-set predictions merged with physical labels.
- `PlatoLightCurves/data/models/cnn_dataset.pt`, `cnn_checkpoint.pt`, `cnn_model.pt`, and `split_indices.pt`: CNN cache/checkpoint artifacts tracked through Git LFS.

The combined label table contains 2,000 simulated systems: 500 quiet, 500 flare-only, 500 transit-only, and 500 flare-plus-transit examples. The drift labels are distributed across `none`, `low`, `medium`, `high`, and `max`.

## Notebook Guide

### `PlatoLightCurves/scripts/data_generation.ipynb`

This notebook defines the labeled synthetic dataset. It loads the PSLS base configuration and stellar grid, samples valid stellar parameters, assigns each system one of the four anomaly classes, and prepares per-system YAML files under `PlatoLightCurves/data/configs/`.

The notebook controls the injected astrophysical signals:

- For flare-positive systems, it enables the PSLS flare model and samples flare amplitude, duration, and timing parameters.
- For transit-positive systems, it enables the transit model and samples planet radius and orbital period.
- For mixed systems, it enables both flare and transit components.
- For quiet systems, both event injectors remain disabled while stellar and instrumental variability are still present.

The first generated batch represents no-drift systems. A second batch introduces drift categories so that the classifiers can be evaluated under harder instrumental conditions. The notebook then combines the clean and drift label tables into `PlatoLightCurves/data/input_labels/paired_simulation_labels_combined.csv`.

It also contains plots and summary checks for the sampled parameter distributions, including stellar mass, effective temperature, surface gravity, flare amplitude, transit radius, and orbital period. Those plots are useful for justifying the scientific design of the simulation.

### `PlatoLightCurves/scripts/run_simulations.py`

This script batch-runs PSLS for the generated YAML configs. For each system ID, it finds the matching config, creates an output directory, runs `psls.py`, and writes logs if the simulator fails.

The script is useful for regenerating light curves, but it should be treated as a batch-production helper rather than the main live-demo target. Before running it in a fresh environment, check that its path constants point to the current repo layout:

- configs should come from `PlatoLightCurves/data/configs/`
- outputs should go to `PlatoLightCurves/data/outputs/`
- labels should come from `PlatoLightCurves/data/input_labels/`
- PSLS should point to `PlatoLightCurves/psls-1.9/psls.py`

### `PlatoLightCurves/scripts/preprocessing.ipynb`

This notebook corrects instrumental drift artifacts in the simulated light curves. Drift matters because slow baseline changes and sudden mask-update jumps can make a transit look shallower, hide a flare, or create false event-like structure.

The notebook performs two main corrections:

- Jump correction: it identifies mask-update locations from the third column of each `.dat` file, compares median flux before and after each jump, and shifts later segments so the baseline remains continuous.
- Slope correction: it estimates a low-frequency baseline with a downsampled Savitzky-Golay filter, interpolates that baseline back to the full cadence, and subtracts it from the light curve.

The notebook includes diagnostic functions that report segment medians, residual slopes, original slopes, and variance reduction. It writes corrected files named `*_driftcorrected.dat` next to the original `.dat` files in `PlatoLightCurves/data/outputs/sys_XXXX/`.

This notebook is the best place to discuss the scientific tradeoff in preprocessing: too little correction leaves instrumental trends in the data, while too much correction can remove real shallow transits or low-amplitude flares.

### `PlatoLightCurves/scripts/CNN.ipynb`

This notebook trains a one-dimensional convolutional neural network directly on light-curve waveforms. It treats the problem as two binary questions rather than one four-class question: does the system contain a flare, and does it contain a transit? The two outputs can represent all four event classes.

The first stage is dataset caching. The notebook loads raw or drift-corrected light curves, removes invalid points, sorts by time, median-centers the flux, normalizes time, interpolates each system to a fixed 50,000-point grid, and saves the result as `PlatoLightCurves/data/models/cnn_dataset.pt`. It also stores labels, system IDs, flare amplitudes, and planet radii so that later analysis can connect model performance to physical parameters.

The model architecture is `LightCurveCNN`. It uses four convolution blocks with batch normalization, ReLU activations, and pooling. The kernels are designed to capture local time-series shapes: sharp flare spikes, transit ingress and egress, and broader waveform patterns. After adaptive pooling, a shared fully connected network feeds two separate heads, one for flares and one for transits.

Training uses Adam, binary cross-entropy style losses, and validation metrics for each head. The notebook reports precision, recall, F1 score, and confusion statistics, then explores threshold tuning. It also includes analysis of detection probability versus flare amplitude and transit radius, which is important because small planets and weak flares are the hardest cases.

### `PlatoLightCurves/scripts/random_forest.ipynb`

This notebook trains the most interpretable classifier in the project. It reads the same cached light curves used by the CNN from `PlatoLightCurves/data/models/cnn_dataset.pt`, extracts hand-crafted features, applies the same saved train/validation split from `split_indices.pt`, fits a scikit-learn random forest, and saves the trained pipeline as `PlatoLightCurves/data/models/rf_checkpoint.joblib`.

The feature extraction step converts each 50,000-point waveform into a compact feature vector. The features include:

- Shape statistics such as skewness, kurtosis, negative-outlier fraction, and robust noise estimates.
- Flare indicators such as high-percentile signal-to-noise, positive flux area, fraction of points above 3-sigma and 5-sigma, longest high-flux run, and local peak contrast.
- Transit indicators from Box Least Squares, including best period, duration, depth, depth signal-to-noise, and peak BLS power.

After training, the notebook prints a confusion matrix and classification report. It then builds `PlatoLightCurves/data/models/rf_full_analysis.csv`, which merges validation predictions with the physical metadata from `paired_simulation_labels_combined.csv`. The later cells use this table to analyze detection probability versus planet radius, flare detection efficiency versus amplitude, accuracy versus drift category, and feature importance.

This is the recommended live-run notebook because it is easier to explain line by line than the neural network while still covering the complete modeling process.

### `PlatoLightCurves/scripts/analyze_classifiers.ipynb`

This notebook is a short follow-up analysis workspace intended for comparing saved classifiers and loading model artifacts. It currently imports the scientific stack and begins from saved `.pt` or `.joblib` outputs rather than rebuilding simulations. It is useful as a lightweight place to add final comparison plots, but it is not the main training notebook.

If more time is available, this notebook should become the final model-comparison report: load the CNN checkpoint, load `rf_checkpoint.joblib`, evaluate both on the same split, compare precision/recall by class, and summarize where each model fails.

## End-to-End Workflow

The project flow is:

```text
PSLS stellar grid and base YAML
        |
        v
light_curves.ipynb
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

## How to Run

Create an environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd PlatoLightCurves/psls-1.9
python setup.py install
cd ../..
```

Open notebooks from `PlatoLightCurves/scripts/` so their relative paths to `../data/` and `../psls-1.9/` resolve correctly.

Recommended live run:

```text
PlatoLightCurves/scripts/random_forest.ipynb
```

For the course checks, run:

```bash
pytest --nbmake --nbmake-timeout=60 PlatoLightCurves/scripts/random_forest.ipynb
nbqa pylint PlatoLightCurves/scripts/random_forest.ipynb
```

If you present a different notebook, replace the path in both commands.

## Presentation Focus

A clear 10-minute presentation can follow this structure:

1. Explain the scientific goal: classify quiet, flare, transit, and mixed PLATO-like light curves.
2. Explain the need: realistic light curves include stellar variability, gaps, and instrumental drift.
3. Walk through the repo layout: `data/`, `psls-1.9/`, and `scripts/`.
4. Live-run `random_forest.ipynb`.
5. Explain one feature-extraction or model-training cell line by line.
6. Discuss the main challenge: weak transits and low-amplitude flares are difficult, especially under drift.
7. Reflect on best practices: fixed labels, saved splits, reusable data artifacts, and separation between generation, preprocessing, and modeling.

## Notes and Current Limitations

- The project uses lowercase `PlatoLightCurves/scripts/` in the current structure.
- Some older notebook cells may still contain paths from before the reorganization. The intended current paths are `../data/input_labels/`, `../data/outputs/`, and `../data/models/` when running from `PlatoLightCurves/scripts/`.
- The `.pt` CNN artifacts are stored through Git LFS. If they appear as small text pointer files after cloning, run `git lfs pull` or regenerate them from `CNN.ipynb`.
- `run_simulations.py` still needs a path cleanup before it is used as a grading entry point. The checked-in light-curve outputs are already under `PlatoLightCurves/data/outputs/`.
- The random forest is the safest notebook for a live demo because it trains quickly, saves an explicit checkpoint, and produces interpretable diagnostics.

