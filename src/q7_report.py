"""Q7 - Reproducibility and reporting.

Collects hardware/software specs, hyperparameters, seeds, epochs, training times, the final
table and the list of figures into results/reproducibility.md (paste into the report).

Run:  python q7_report.py
"""
import glob
import os
import platform
import sys

from common import FIG_DIR, RES_DIR, banner, load_json, save_json, save_text


def hardware():
    info = dict(python=sys.version.split()[0], platform=platform.platform(), processor=platform.processor() or platform.machine())
    try:
        import psutil
        info["cpu_cores_logical"] = psutil.cpu_count()
        info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        info["cpu_cores_logical"] = os.cpu_count()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    for mod in ("numpy", "pandas", "matplotlib", "h5py"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    return info


if __name__ == "__main__":
    banner("Q7  Reproducibility summary")
    hw = hardware()
    for k, v in hw.items():
        print(f"  {k:20s}: {v}")
    lines = ["# Reproducibility summary (AI651 PA3)", "", "## Hardware / software"]
    lines += [f"- **{k}**: {v}" for k, v in hw.items()]

    lines += ["", "## Data", ]
    q3 = load_json("q3_summary.json", {}) or {}
    q4 = load_json("q4_summary.json", {}) or {}
    a = q3.get("a", {})
    lines += [f"- LargeST San Diego (District 11), {a.get('N', 716)} sensors, 2020-2021, 5-min -> 15-min "
              f"({a.get('aggregation', 'sum')} of three 5-min counts; unit: {a.get('unit', 'vehicles / 15 min')})",
              f"- Missing 15-min values: {100*a['missing_15min']:.2f}%" if "missing_15min" in a else "- Missing values: see Q3",
              "- Missing targets masked in loss and metrics; inputs forward-filled (causal), leading gaps = 2020 sensor mean",
              "- Splits: train 2020, val Jan-Mar 2021, test Apr-Dec 2021; z-score with 2020 statistics only",
              "- Input 8 steps (2 h) -> output 4 steps (1 h); windows never cross split boundaries"]
    w = q4.get("windows", {})
    if w:
        lines += [f"- Windows: train {w.get('n_train_windows')}, val {w.get('n_val_windows')}, test {w.get('n_test_windows')}"]

    lines += ["", "## Models, hyperparameters and training"]
    hdr = "| Model | Adjacency | Params | Epochs run | Best epoch | s/epoch | Train time (min) | Test MAE | Test RMSE |"
    lines += [hdr, "|" + "---|" * 9]
    hp = None
    for key, adj in [("temporal_only", "None"), ("stgnn_fixed", "Fixed"), ("stgnn_adaptive", "Adaptive"), ("stgnn_hybrid", "Hybrid")]:
        r = load_json(f"metrics_{key}.json")
        if r is None:
            continue
        hp = r["hparams"]
        lines.append(f"| {'Temporal-only' if key == 'temporal_only' else 'STGNN'} | {adj} | {r['params']:,} | {r['epochs_run']} | "
                     f"{r['best_epoch']} | {r['mean_epoch_time_s']:.1f} | {r['train_time_s']/60:.2f} | "
                     f"{r['test']['MAE']:.3f} | {r['test']['RMSE']:.3f} |")
        if key == "stgnn_hybrid" and "alpha" in r:
            alpha = r["alpha"]
    if hp:
        lines += ["", "Shared hyperparameters (identical for all models):", ""]
        lines += [f"- {k}: {v}" for k, v in hp.items()]
        lines += ["- Temporal encoder: Conv1d(1->d,k=3) -> ReLU -> Conv1d(d->d,k=3,dilation=2) -> ReLU -> Linear(8d->d) -> ReLU",
                  "- Graph conv: H = ReLU(A_hat Z W_g), W_g in R^{d x d}",
                  "- Head: Linear(d->d) -> ReLU -> Linear(d->4)",
                  "- Fixed graph: D^-1(A+I) if A is directed, else D^-1/2(A+I)D^-1/2",
                  "- Adaptive graph: softmax_row(ReLU(E1 E2^T)), E1,E2 ~ N(0,1) init",
                  "- Hybrid: alpha*D^-1(A+I) + (1-alpha)*A_adaptive, alpha = sigmoid(learned scalar), init 0.5",
                  "- Early stopping on validation MAE; best checkpoint evaluated on test"]
    try:
        lines.append(f"- Final learned alpha: {alpha:.4f}")
    except NameError:
        pass
    lines += ["", "## Figures", ""] + [f"- {os.path.basename(p)}" for p in sorted(glob.glob(os.path.join(FIG_DIR, '*.png')))]
    lines += ["", "## How to reproduce", "",
              "```", "pip install -r requirements.txt",
              "export PA3_DATA_DIR=<folder for LargeST files>  PA3_OUT_DIR=<output folder>",
              "python q1_aggregation.py", "python q2_direction.py", "python q3_dataset.py",
              "python q4_stgnn.py --epochs 30 --hidden 32 --seed 42",
              "python q5_adjacency.py --epochs 30 --hidden 32 --seed 42",
              "python q6_analysis.py", "python q7_report.py", "```"]
    text = "\n".join(lines)
    save_text(text, "reproducibility.md")
    save_json(hw, "hardware.json")
    print("\n" + text)
    print("\nDone with Q7.")
