# AI651 — Programming Assignment 3

**Deep Learning for Space, Time and Graphs | Fall 2026**  
**Lahore University of Management Sciences (LUMS)**  
**Author:** Mubeen Ahmed · **Student ID:** 25280101

## Overview

This repository documents Programming Assignment 3 for AI651. It examines how graph-message aggregation affects expressivity, why direction matters in a moving traffic pattern, and whether graph structure improves spatio-temporal forecasting on the **LargeST San Diego (SD)** subset.

The assignment is organised into seven questions:

| Question | Topic | Main implementation / output |
|---|---|---|
| Q1 | Sum, mean and max aggregation | Analytical examples and exhaustive verification of count encoding |
| Q2 | Direction of slowdown on a 40-sensor ring | Least-squares fitting, exact directed/history rules and rollout heatmaps |
| Q3 | LargeST-SD data preparation and road graph | Resampling, missingness, adjacency and neighbourhood visualisations |
| Q4 | Spatio-temporal graph forecasting | Temporal-only baseline and fixed-adjacency STGNN |
| Q5 | Fixed, adaptive and hybrid adjacency | Trainable node embeddings, learned/hybrid graphs and comparisons |
| Q6 | Forecast and graph interpretation | Final metrics, predicted-versus-actual plots and graph analysis |
| Q7 | Reproducibility | Configuration, runtime environment, measurements and reporting |

## Repository structure

```text
AI651-PA3/
├── src/                            # Add original files from pa3_code.zip
│   ├── common.py
│   ├── data_utils.py
│   ├── largest_io.py
│   ├── models.py
│   ├── q1_aggregation.py
│   ├── q2_direction.py
│   ├── q3_dataset.py
│   ├── q4_stgnn.py
│   ├── q5_adjacency.py
│   ├── q6_analysis.py
│   ├── q7_report.py
│   └── requirements.txt
├── notebooks/
│   └── PA3_runner.ipynb
├── outputs/
│   ├── figures/                    # Saved experimental plots (PNG/PDF)
│   ├── results/                    # Metrics, CSV, JSON, arrays and predictions             
├── report/
│   ├── main.tex
│   ├── main.pdf
│   ├── tables/
└── figures/                    
```

**Important:** The supplied files for this repository starter contain the executed notebook and saved experimental outputs.
## Dataset and forecasting task

- **Dataset:** [LargeST](https://github.com/liuxu77/LargeST), San Diego subset (District 11), **716 traffic sensors**.
- **Raw data:** 2020 and 2021, sampled at **5-minute intervals**.
- **Resampling used:** sum each three consecutive 5-minute vehicle-count readings to obtain a **15-minute count**, in **vehicles per 15 minutes**.
- **Forecasting windows:** 8 observed intervals (2 hours) to predict the next 4 intervals (1 hour), for all 716 sensors.
- **Chronological split:** 2020 training; January–March 2021 validation; April–December 2021 testing.
- **Normalization:** training-period statistics only; no forecasting window crosses a split boundary.
- **Missing data:** causal forward filling of inputs, training-period mean for leading gaps, and missing-target masks in the objective/evaluation. Some sensors are entirely missing; the precise fallback implementation should be verified against the original source code.

**Data access:** [Kaggle — LargeST dataset](https://www.kaggle.com/datasets/liuxu77/largest). Raw and processed datasets are not committed to this repository. The runner expects the LargeST California metadata/adjacency and 2020–2021 raw traffic files. Kaggle credentials (`kaggle.json`) must **never** be committed.

## Quick start: Google Colab (recommended)

The provided **`notebooks/PA3_runner.ipynb`** is the actual experiment runner used for the saved outputs. Its cells mount Google Drive, upload `pa3_code.zip`, configure training, and run Q1–Q7 in sequence.

1. Put the original scripts and `requirements.txt` in `src/`. Create the archive expected by the notebook:

   ```bash
   cd src
   zip -r ../pa3_code.zip . -x '__pycache__/*'
   cd ..
   ```

   Upload `pa3_code.zip` when prompted by the notebook. **Do not commit** this generated archive to Git; the scripts should already be version-controlled in `src/`.

2. Open `notebooks/PA3_runner.ipynb` in Google Colab and select a **T4 GPU** runtime (or another available GPU).
3. Run cells **from top to bottom**. Grant Google Drive access and upload `pa3_code.zip` in cell 2.
4. If the processed SD dataset is not already cached, supply Kaggle API credentials (`kaggle.json`) when prompted. The notebook caches the processed data on Drive.
5. Run Q1–Q7 and use the final cell to export figures and results.

The notebook's default Drive folders are:

```text
/content/drive/MyDrive/AI651_PA3/
├── data/               # Cached, preprocessed dataset
├── outputs/
│   ├── figures/
│   ├── results/
│   └── ...              # Training checkpoints
└── pa3_code.zip         # Uploaded source archive copy
```
