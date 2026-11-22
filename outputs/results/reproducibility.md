# Reproducibility summary (AI651 PA3)

## Hardware / software
- **python**: 3.13.16
- **platform**: Linux-6.6.159+-x86_64-with-glibc2.39
- **processor**: x86_64
- **cpu_cores_logical**: 2
- **cpu_cores_physical**: 1
- **ram_gb**: 13.6
- **cpu_model**: Intel(R) Xeon(R) CPU @ 2.00GHz
- **torch**: 2.14.1+cu130
- **cuda_available**: True
- **gpu**: Tesla T4
- **gpu_mem_gb**: 15.6
- **cuda**: 13.0
- **numpy**: 2.4.6
- **pandas**: 2.2.3
- **matplotlib**: 3.10.0
- **h5py**: 3.14.0

## Data
- LargeST San Diego (District 11), 716 sensors, 2020-2021, 5-min -> 15-min (sum of three 5-min counts; unit: vehicles / 15 min)
- Missing 15-min values: 5.06%
- Missing targets masked in loss and metrics; inputs forward-filled (causal), leading gaps = 2020 sensor mean
- Splits: train 2020, val Jan-Mar 2021, test Apr-Dec 2021; z-score with 2020 statistics only
- Input 8 steps (2 h) -> output 4 steps (1 h); windows never cross split boundaries
- Windows: train 35125, val 8629, test 26389

## Models, hyperparameters and training
| Model | Adjacency | Params | Epochs run | Best epoch | s/epoch | Train time (min) | Test MAE | Test RMSE |
|---|---|---|---|---|---|---|---|---|
| Temporal-only | None | 12,644 | 30 | 29 | 16.3 | 8.15 | 64.552 | 101.312 |
| STGNN | Fixed | 13,668 | 30 | 29 | 17.2 | 8.60 | 140.232 | 218.190 |
| STGNN | Adaptive | 27,988 | 30 | 29 | 17.7 | 8.85 | 82.169 | 129.802 |
| STGNN | Hybrid | 27,989 | 30 | 28 | 17.7 | 8.87 | 79.310 | 125.542 |

Shared hyperparameters (identical for all models):

- epochs: 30
- patience: 6
- batch_size: 64
- lr: 0.001
- weight_decay: 0.0
- hidden: 32
- emb_dim: 10
- topk: 0
- alpha: None
- train_stride: 1
- seed: 42
- in_len: 8
- out_len: 4
- loss: masked MAE
- optimizer: Adam
- grad_clip: 5.0
- Temporal encoder: Conv1d(1->d,k=3) -> ReLU -> Conv1d(d->d,k=3,dilation=2) -> ReLU -> Linear(8d->d) -> ReLU
- Graph conv: H = ReLU(A_hat Z W_g), W_g in R^{d x d}
- Head: Linear(d->d) -> ReLU -> Linear(d->4)
- Fixed graph: D^-1(A+I) if A is directed, else D^-1/2(A+I)D^-1/2
- Adaptive graph: softmax_row(ReLU(E1 E2^T)), E1,E2 ~ N(0,1) init
- Hybrid: alpha*D^-1(A+I) + (1-alpha)*A_adaptive, alpha = sigmoid(learned scalar), init 0.5
- Early stopping on validation MAE; best checkpoint evaluated on test
- Final learned alpha: 0.2580

## Figures

- q1b_encoding_injective.png
- q2a_heatmaps.png
- q2a_profiles_t20.png
- q2b_heatmaps.png
- q2c_heatmaps.png
- q3a_missing_values.png
- q3a_three_sensors_one_week.png
- q3b_adjacency_heatmap.png
- q3b_degree_distribution.png
- q3c_neighbourhood_map.png
- q3c_neighbourhood_zoom.png
- q3c_receptive_field_growth.png
- q4a_splits.png
- q4c_temporal_vs_graph.png
- q5_adjacency_matrices.png
- q5_learning_curves.png
- q5b_learned_graph_rows.png
- q6a_best_model_one_day_all_horizons.png
- q6a_metric_bars.png
- q6a_pred_vs_actual_15min.png
- q6a_pred_vs_actual_60min.png
- q6b_correlation_of_learned_edges.png
- q6b_distance_distribution.png
- q6b_learned_edges_map.png
- q6b_road_vs_learned_sorted.png

## How to reproduce

```
pip install -r requirements.txt
export PA3_DATA_DIR=<folder for LargeST files>  PA3_OUT_DIR=<output folder>
python q1_aggregation.py
python q2_direction.py
python q3_dataset.py
python q4_stgnn.py --epochs 30 --hidden 32 --seed 42
python q5_adjacency.py --epochs 30 --hidden 32 --seed 42
python q6_analysis.py
python q7_report.py
```