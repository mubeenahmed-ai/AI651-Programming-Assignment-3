"""Shared helpers for AI651 PA3 (paths, seeding, figure saving, JSON I/O).

Paths are controlled by environment variables so the same scripts run locally
and on Colab (where the notebook points them at Google Drive):

    PA3_DATA_DIR : raw LargeST files + processed San Diego arrays (big, cached)
    PA3_OUT_DIR  : figures, tables, metrics, checkpoints (for the report)
"""
import json
import os
import random

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42

DATA_DIR = os.environ.get("PA3_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
OUT_DIR = os.environ.get("PA3_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
RAW_DIR = os.environ.get("PA3_RAW_DIR", os.path.join(DATA_DIR, "raw"))  # can be local disk (fast)
PROC_DIR = os.path.join(DATA_DIR, "processed")
FIG_DIR = os.path.join(OUT_DIR, "figures")
RES_DIR = os.path.join(OUT_DIR, "results")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

for _d in (RAW_DIR, PROC_DIR, FIG_DIR, RES_DIR, CKPT_DIR):
    os.makedirs(_d, exist_ok=True)

PROCESSED_FILE = os.path.join(PROC_DIR, "sd_flow15.npz")
META_FILE = os.path.join(PROC_DIR, "sd_meta.csv")


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def savefig(fig, name: str, dpi: int = 150):
    """Save a figure as PNG (for the notebook) and PDF (for the report)."""
    png = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved {png}")
    return png


def save_json(obj, name: str):
    path = os.path.join(RES_DIR, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)  # atomic (safe on Drive)
    print(f"[json] saved {path}")
    return path


def load_json(name: str, default=None):
    path = os.path.join(RES_DIR, name)
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def save_text(text: str, name: str):
    path = os.path.join(RES_DIR, name)
    with open(path, "w") as f:
        f.write(text)
    print(f"[txt] saved {path}")
    return path


def banner(title: str):
    line = "=" * max(60, len(title) + 4)
    print(f"\n{line}\n  {title}\n{line}")


def to_datetime_index(arr):
    """Integer epoch array -> DatetimeIndex, inferring the unit (s/ms/us/ns) from magnitude
    (robust to pandas 2 vs 3, which store datetimes with different default resolutions)."""
    import pandas as pd
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.integer):
        return pd.DatetimeIndex(pd.to_datetime(arr))
    mx = np.abs(arr).max() if arr.size else 0
    unit = "ns" if mx > 1e17 else "us" if mx > 1e14 else "ms" if mx > 1e11 else "s"
    return pd.DatetimeIndex(pd.to_datetime(arr, unit=unit))


def datetime_to_ns(ts):
    """DatetimeIndex -> int64 nanoseconds since epoch."""
    return np.asarray(ts, dtype="datetime64[ns]").astype(np.int64)
