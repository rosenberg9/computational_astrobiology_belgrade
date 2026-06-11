# PLATO Light Curve Classification Pipeline

This repository implements a machine learning pipeline for classifying synthetic PLATO mission light curves into four categories: stellar flares, planetary transits, both, or neither. Simulated light curves are generated using [PSLS v1.9](https://gitlab.com/plato-mission/psls) and classified using both a 1D Convolutional Neural Network and a Random Forest trained on hand-crafted features.

---

## Repository Structure

```
.
├── data_generation.ipynb              # Simulate light curves via PSLS
├── preprocessing.ipynb                # Instrument drift correction
├── CNN.ipynb                          # CNN classifier
├── random_forest.ipynb                # Random Forest classifier
├── paired_simulation_labels_combined.csv  # Master labels file
├── configs/                           # Per-system PSLS YAML configs
├── outputs/                           # Simulated .dat light curve files
├── cnn_dataset.pt                     # Pre-baked tensor cache (generated)
└── cnn_checkpoint.pt                  # Trained CNN weights (generated)
```

---

## Notebooks

### 1. `data_generation.ipynb` — Simulated Dataset Generation

Generates a labelled dataset of synthetic PLATO light curves by driving PSLS with randomised astrophysical parameters. Two batches are produced:

- **~1,000 systems** with no instrumental drift (`drift = "none"`)
- **~490 systems** with varying drift levels (`low`, `medium`, `high`, `max`)

Each system is assigned one of four anomaly classes:

| Class | Label      | Description                    |
|-------|------------|--------------------------------|
| 0     | Neither    | Quiet stellar photometry       |
| 1     | Flare      | Stellar flare injected         |
| 2     | Transit    | Planetary transit injected     |
| 3     | Both       | Flare and transit both present |

**Stellar parameters** are sampled from the PSLS pre-computed stellar grid (`grid_plato.hdf5`), which covers solar-like stars. **Flare parameters** are drawn log-uniformly: amplitude in [500, 20,000] ppm, mean period in [1, 15] days, mean duration in [0.01, 0.1] days. **Transit parameters** are drawn from a log-normal planet radius prior centred at 1.7 R⊕ (σ = 1.0) and a log-uniform orbital period in [2, 100] days, with semi-major axis computed via Kepler's third law.

The notebook also includes visualisations of the injected priors (corner plots of stellar parameters, histograms of transit and flare distributions) and combines both batches into `paired_simulation_labels_combined.csv`.

**Key outputs:** `configs/sys_XXXX.yaml`, `paired_simulation_labels_combined.csv`

---

### 2. `preprocessing.ipynb` — Instrument Drift Correction

The PLATO instrument periodically realigns its pointing to correct for thermo-elastic drift, which introduces two artefacts: discrete flux jumps at each mask update, and a slow intra-segment slope. This notebook corrects both.

**Jump correction** identifies the indices of mask update events from the flag column of each `.dat` file, then aligns adjacent segments by shifting the post-jump flux so that median levels match across the boundary.

**Slope correction** fits a low-frequency Savitzky-Golay baseline to each segment using a downsampled version of the light curve (every 100th point) for speed, interpolates the baseline back to full resolution, and subtracts it.

The pipeline processes all systems in `paired_simulation_labels_combined.csv` that carry a non-zero drift label and writes corrected files as `*_driftcorrected.dat` alongside the originals. A segment-by-segment diagnostic function reports residual median, slope, and variance reduction for quality control.

**Key outputs:** `outputs/sys_XXXX/*_driftcorrected.dat`

---

### 3. `CNN.ipynb` — 1D Convolutional Neural Network Classifier

Trains a 1D CNN on the full light curve waveform, treating classification as a multi-label problem with two independent binary heads: `head_flare` and `head_transit`.

**Pre-baking** (`build_prebaked_dataset`) loads each raw light curve, median-centres the flux, normalises time to [0, 1], interpolates to a uniform 50,000-point grid, and saves the result as a PyTorch tensor cache (`cnn_dataset.pt`) together with flare amplitudes, planet radii, and system IDs. This step is run once; subsequent training loads directly from the cache.

**Architecture** (`LightCurveCNN`): four 1D convolutional layers with BatchNorm, ReLU, and MaxPool (reducing 50k → 12.5k → 3,125 → 781 → 16 via adaptive pooling), followed by shared fully-connected layers (2048 → 256 → 64) with dropout, then two independent linear heads. The design uses progressively smaller kernels (15 → 11 → 7 → 5) to capture both transit-scale slopes and sharp flare spikes.

**Loss functions**: `BCEWithLogitsLoss` applied separately to each head. A `transit_loss` wrapper is included that supports radius-weighted penalties to reduce missed detections of large planets.

**Training** runs for 30 epochs with Adam (lr = 4×10⁻⁴, weight decay = 10⁻⁴), reporting per-epoch precision, recall, F1, and confusion statistics for each head separately.

**Post-training analysis** includes detection probability curves as a function of flare amplitude and planet radius, adjustable classification thresholds (default: 0.5 for flares, tuned around 0.28–0.35 for transits), and light curve visualisations of difficult cases (small planets, faint flares).

**Key outputs:** `cnn_dataset.pt`, `cnn_checkpoint.pt`

---

### 4. `random_forest.ipynb` — Random Forest Classifier with Hand-Crafted Features

Trains a scikit-learn Random Forest on a feature vector derived from each light curve, providing an interpretable baseline and complementary diagnostics to the CNN.

**Feature extraction** is parallelised over CPU cores with `ThreadPoolExecutor`. Three feature groups are computed per system:

- **Statistical / noise features** (16 features): point count, time span, cadence, flux standard deviation, MAD-based noise estimate (σ_MAD), skewness, kurtosis, percentiles (P1–P99), tail widths, and negative-outlier fraction.
- **Flare features** (9 features): peak SNR, P99 SNR, positive flux area, fraction of points above 3σ and 5σ, longest contiguous above-3σ run (and its duration in days), and local peak contrast around the flux maximum.
- **BLS transit features** (6 features): peak BLS power, best-fit period, duration, transit depth in ppm, depth SNR, and transit time. BLS searches log-uniformly spaced periods from 2 to 100 days with durations of 0.08, 0.16, 0.32, and 0.64 days.

**Model**: a scikit-learn `Pipeline` with a median imputer (for NaN BLS values on short baselines) and a `RandomForestClassifier` (300 trees, balanced class weights, all CPU cores). Trained on a 75/25 stratified split.

**Analysis** covers confusion matrix, classification report, feature importance ranking, detection probability vs. planet radius (overall and restricted to no-drift systems), detection efficiency vs. flare amplitude, accuracy stratified by drift level, and case studies of failed detections including BLS periodogram diagnostics for individual systems.

---

## Data Flow

```
PSLS stellar grid (grid_plato.hdf5)
        │
        ▼
data_generation.ipynb  ──►  configs/sys_XXXX.yaml
                       ──►  outputs/sys_XXXX/*.dat
                       ──►  paired_simulation_labels_combined.csv
        │
        ▼
preprocessing.ipynb    ──►  outputs/sys_XXXX/*_driftcorrected.dat
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
CNN.ipynb                              random_forest.ipynb
  └► cnn_dataset.pt                      └► feature matrix
  └► cnn_checkpoint.pt                   └► trained pipeline
```

---

## Dependencies

```
torch, numpy, pandas, scipy, sklearn, astropy, matplotlib, seaborn, tqdm, h5py, pyyaml, corner
```

PSLS v1.9 must be installed separately and its `psls.yaml` and `grid_plato.hdf5` files placed at `../psls-1.9/` relative to this directory.

---

## Notes

- Large data files (`outputs/`, `cnn_dataset.pt`, `*.dat`) are excluded from version control via `.gitignore`. The `configs/` directory and label CSVs are tracked.
- The CNN and Random Forest operate on the same underlying dataset but differ in their input representation: the CNN sees the raw interpolated waveform; the Random Forest sees a 31-dimensional feature vector per system.
- Transit recall is the primary challenge, particularly for planets below ~2 R⊕. The `transit_loss` radius-weighting term in the CNN and the BLS-derived features in the Random Forest are both targeted at this sensitivity limit.