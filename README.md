# Physics-Informed Faraday Cup Design Optimizer

A physics-based Monte Carlo simulation and machine learning surrogate modeling project for estimating secondary-electron plate-hopping risk in a segmented Faraday cup collector.

This repository builds an end-to-end workflow:

```text
physics simulation -> Monte Carlo dataset generation -> supervised ML surrogate -> fast design-space prediction
```

The goal is to predict when secondary electrons emitted from a collector surface return to the same plate, hop to another plate, hit a gap, strike the suppressor grid, or escape through a grid opening.

---

## Motivation

Faraday cups measure charged particle flux by collecting ions on segmented plates. When incoming ions strike the collector surface, they can emit secondary electrons. Some of these secondary electrons may move laterally and land on a different collector quadrant.

This project asks:

> For a given Faraday cup design and operating condition, what fraction of incoming ions produce plate-hopping secondary electrons?

This matters because plate hopping can distort quadrant-level current measurements and affect the interpretation of plasma instrument data.

---

## Project Overview

The repository implements a full physics-informed machine learning pipeline:

1. Sample incoming proton impacts.
2. Sample secondary-electron emission counts, energies, and directions.
3. Solve an approximate wire-grid suppressor electric field.
4. Integrate electron trajectories under electric and magnetic fields.
5. Classify each electron outcome.
6. Pool Monte Carlo results into design-level labels.
7. Generate train/test datasets over physical design parameters.
8. Train surrogate ML models to predict hopping probability.
9. Evaluate models against a baseline and save metrics, predictions, and plots.

The machine learning portion treats the simulator as an expensive data-generating process. The surrogate model learns a fast approximation from design parameters to hopping risk.

---

## Core Prediction Target

The primary supervised-learning target is:

```text
ion_hopping_probability
```

This is the fraction of valid incoming ion impacts that produce at least one secondary electron that lands on a different collector quadrant.

Secondary targets include:

```text
electron_hopping_probability
return_probability
```

---

## Input Features

The surrogate model uses physical design and operating parameters as inputs.

### Geometry Features

```text
gap_width_mm
grid_spacing_mm
grid_height_mm
wire_radius_mm
```

### Operating Features

```text
proton_energy_eV
incidence_angle_deg
suppressor_voltage
```

### Magnetic Field Features

```text
magnetic_field_x_nT
magnetic_field_y_nT
magnetic_field_z_nT
magnetic_field_magnitude_nT
```

### Engineered Physics Features

```text
grid_spacing_over_height
wire_radius_over_spacing
gap_width_over_grid_spacing
field_strength_proxy
```

Monte Carlo output columns are intentionally excluded from the model inputs to avoid label leakage.

---

## Physics Simulation

The simulator models the following components:

- A segmented four-quadrant collector.
- A cross-shaped inter-plate gap.
- Proton impact sampling over the collector surface.
- Secondary-electron emission using empirical yield data.
- Secondary-electron energy and angular sampling.
- A wire-grid suppressor electric-field approximation.
- Uniform magnetic-field effects through Lorentz-force dynamics.
- Event detection for collector return, suppressor-grid interaction, wire strike, radial escape, timeout, and solver failure.

Electron outcomes are classified as:

```text
same_quadrant
different_quadrant
gap
outside
hit_wire
passed_grid_opening
did_not_return
solver_failure
```

The key outcome for ML is `different_quadrant`, because that represents a plate-hopping secondary electron.

---

## Preliminary Physics Result

A suppressor-voltage sweep without magnetic field showed the expected trend: stronger negative suppressor voltage reduced hopping and increased return probability.


| Suppressor Voltage | Ion Hopping Probability | Electron Hopping Probability | Return Probability |
| ------------------ | ----------------------- | ---------------------------- | ------------------ |
| -20 V              | 1.70%                   | 12.79%                       | 83.32%             |
| -35 V              | 1.43%                   | 10.87%                       | 89.23%             |
| -55 V              | 1.00%                   | 7.54%                        | 92.13%             |
| -75 V              | 0.83%                   | 6.28%                        | 94.12%             |
| -100 V             | 0.66%                   | 4.99%                        | 95.28%             |


This matched the expected physics: a stronger negative suppressor field turns secondary electrons around faster, reducing lateral displacement and lowering plate-hopping probability.

The production dataset generator extends this by sampling geometry, ion energy, incidence angle, suppressor voltage, and magnetic-field components.

---

## Repository Structure

```text
Faraday-Hopping-Surrogate/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
│   ├── figures/
│   └── metrics/
├── scripts/
│   ├── run_small_monte_carlo.py
│   ├── sweep_suppressor_voltage.py
│   ├── plot_suppressor_voltage_sweep.py
│   ├── generate_surrogate_dataset.py
│   └── generate_train_test_surrogate_datasets.py
├── src/
│   ├── physics/
│   │   ├── constants.py
│   │   ├── emission.py
│   │   ├── fields.py
│   │   ├── geometry.py
│   │   ├── impact.py
│   │   └── trajectory.py
│   ├── simulation/
│   │   ├── labels.py
│   │   └── monte_carlo.py
│   └── ml/
│       ├── features.py
│       ├── metrics.py
│       ├── models.py
│       ├── plotting.py
│       └── train_surrogate_model.py
└── tests/
```

---

## Setup

This project uses Python and `uv`.

```bash
git clone https://github.com/ctrl-is/Faraday-Hopping-Surrogate.git
cd Faraday-Hopping-Surrogate
uv sync
```

Run tests:

```bash
PYTHONPATH=. uv run pytest -v
```

Run linting and formatting:

```bash
uvx ruff check . --fix
uvx ruff format .
uvx ruff check .
uvx ruff format --check .
```

---

## Running the Physics Simulation

Run a small end-to-end Monte Carlo smoke test:

```bash
PYTHONPATH=. uv run python scripts/run_small_monte_carlo.py
```

Run the suppressor-voltage sweep:

```bash
PYTHONPATH=. uv run python scripts/sweep_suppressor_voltage.py
```

Plot the voltage sweep results:

```bash
PYTHONPATH=. uv run python scripts/plot_suppressor_voltage_sweep.py
```

---

## Generating Train/Test Data

The train/test dataset generator creates separate CSV files:

```text
data/processed/surrogate_train.csv
data/processed/surrogate_test.csv
```

Run:

```bash
PYTHONPATH=. uv run python -u scripts/generate_train_test_surrogate_datasets.py 2>&1 | tee dataset_generation.log
```

The generator samples unique physical designs using Latin hypercube sampling. Each design is evaluated through multiple Monte Carlo seeds, and the counts are pooled into one supervised-learning row.

The train/test split is done by design configuration, not by random seed. This avoids leaking the same physical setup into both train and test sets.

Current production setting:

```text
TRAIN_DESIGNS = 1000
TEST_DESIGNS = 200
NUM_PROTONS_PER_SEED = 5000
SEEDS_PER_DESIGN = 3
```

This produces:

```text
Train: 1000 designs × 3 seeds × 5000 protons = 15,000,000 simulated protons
Test:   200 designs × 3 seeds × 5000 protons =  3,000,000 simulated protons
```

---

## Dataset Format

Each row represents one physical design configuration.

Input columns include:

```text
gap_width_mm
proton_energy_eV
incidence_angle_deg
grid_spacing_mm
grid_height_mm
wire_radius_mm
suppressor_voltage
magnetic_field_x_nT
magnetic_field_y_nT
magnetic_field_z_nT
magnetic_field_magnitude_nT
grid_spacing_over_height
wire_radius_over_spacing
gap_width_over_grid_spacing
field_strength_proxy
```

Label columns include:

```text
ion_hopping_probability
electron_hopping_probability
return_probability
```

The dataset also stores raw Monte Carlo counts for analysis and quality checks, but those count columns are not used as model inputs.

---

## Machine Learning Pipeline

The ML pipeline is organized around reusable utilities:

```text
src/ml/features.py
src/ml/metrics.py
src/ml/models.py
src/ml/plotting.py
```

The training script will:

1. Load train and test CSVs.
2. Select only allowed physical input features.
3. Train a mean-prediction baseline.
4. Train supervised regression models.
5. Evaluate models on the held-out test set.
6. Save the best model.
7. Save metrics, predictions, and plots.

Primary training target:

```text
ion_hopping_probability
```

Planned model comparison:

```text
mean baseline
linear regression
random forest regressor
extra trees regressor
histogram gradient boosting regressor
```

---

## Evaluation Metrics

The model is evaluated using:

```text
MAE
RMSE
R²
max error
improvement over mean baseline
```

For this project, MAE is the most interpretable metric. For example:

```text
MAE = 0.003
```

means the model is off by about `0.3 percentage points` in predicted ion-level hopping probability.

The model is also compared against a mean-prediction baseline to verify that it is learning meaningful structure from the physical design parameters.

---

## Expected ML Outputs

Training the surrogate model will produce:

```text
models/ion_hopping_surrogate.joblib
reports/metrics/ion_hopping_metrics.json
data/predictions/ion_hopping_test_predictions.csv
reports/figures/ion_hopping_predicted_vs_actual.png
reports/figures/residuals_vs_actual_ion_hopping.png
```

---

## Why This Project Is ML-Relevant

This project demonstrates a practical surrogate-modeling workflow:

```text
expensive simulation -> synthetic labeled dataset -> fast supervised surrogate
```

The surrogate model learns to approximate a Monte Carlo trajectory simulator. Instead of running thousands of particle trajectories for every new design, the trained model can quickly estimate hopping risk from physical parameters.

This is relevant to:

- scientific machine learning
- simulation acceleration
- physics-informed feature engineering
- design-space exploration
- uncertainty-aware modeling
- instrument design optimization

---

## Current Status

Implemented:

- Collector geometry and quadrant classification.
- Secondary-electron emission sampling.
- Empirical yield interpolation.
- Wire-grid suppressor electric-field solver.
- Lorentz-force trajectory integration.
- Trajectory event classification.
- Monte Carlo hopping probability estimator.
- Suppressor-voltage sweep experiment.
- Train/test dataset generator.
- ML feature and metric utilities.

In progress:

- Model registry.
- Training script.
- Model comparison.
- Feature importance analysis.
- Model card.

---

## Project Summary

This repository builds a physics-informed ML system for predicting secondary-electron plate-hopping risk in a Faraday cup. It combines particle simulation, Monte Carlo estimation, train/test dataset generation, supervised regression, and model evaluation into a reproducible scientific machine learning workflow.