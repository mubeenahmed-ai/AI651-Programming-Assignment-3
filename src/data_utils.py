"""Forecasting windows, chronological splits, train-only normalisation, adjacency normalisation.

Windows:  X_t = [x_{t-7}, ..., x_t]  (8 x 716),   Y_t = [x_{t+1}, ..., x_{t+4}]  (4 x 716)
Splits :  train = 2020, val = Jan-Mar 2021, test = Apr-Dec 2021. A window is kept only if all
          12 of its timestamps lie in the same split, so no window crosses a split boundary.
"""
import numpy as np
import pandas as pd
import torch

from common import PROCESSED_FILE, META_FILE, to_datetime_index

IN_LEN, OUT_LEN = 8, 4


def split_labels(ts):
    lab = np.full(len(ts), -1, dtype=np.int8)
    lab[ts.year == 2020] = 0
    lab[(ts.year == 2021) & (ts.month <= 3)] = 1
    lab[(ts.year == 2021) & (ts.month >= 4)] = 2
    return lab


def normalize_adj(A, mode="auto"):
    """Add self-loops, then normalise.
    sym : D^-1/2 (A+I) D^-1/2   (GCN, for symmetric non-negative A)
    row : D^-1 (A+I)            (random-walk; correct for DIRECTED graphs, rows sum to 1)
    auto: sym if A is symmetric, else row."""
    A = np.asarray(A, dtype=np.float64)
    if mode == "auto":
        mode = "sym" if np.allclose(A, A.T) else "row"
    At = A.copy()
    np.fill_diagonal(At, 1.0)  # self-loop weight 1 (LargeST already has diag = 1 -> unchanged)
    d = At.sum(1)
    if mode == "sym":
        dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        An = dinv[:, None] * At * dinv[None, :]
    else:
        An = At / np.maximum(d, 1e-12)[:, None]
    return An.astype(np.float32), mode


class TrafficData:
    """Holds the whole 15-min tensor on the target device and serves windows by start index."""

    def __init__(self, device="cpu", train_stride=1):
        d = np.load(PROCESSED_FILE)
        flow = d["flow"].astype(np.float32)                    # (T, N) with NaN = missing
        self.ts = to_datetime_index(d["timestamps"])
        self.ids = d["sensor_ids"]
        self.adj = d["adj"].astype(np.float32)
        self.meta = pd.read_csv(META_FILE)
        self.agg = str(d["agg"]) if "agg" in d else "sum"
        self.T, self.N = flow.shape
        self.split = split_labels(self.ts)
        train = self.split == 0

        mask = ~np.isnan(flow)
        # ---- input imputation: causal forward fill, then TRAIN per-sensor mean
        filled = pd.DataFrame(flow).ffill().to_numpy(np.float32)
        tr_mean_sensor = np.nanmean(np.where(mask[train], flow[train], np.nan), 0)
        glob = float(np.nanmean(flow[train]))
        tr_mean_sensor = np.where(np.isnan(tr_mean_sensor), glob, tr_mean_sensor)
        nan_left = np.isnan(filled)
        filled[nan_left] = np.take(tr_mean_sensor, np.nonzero(nan_left)[1])

        # ---- z-score with TRAINING statistics only (observed values)
        self.mean = float(flow[train][mask[train]].mean())
        self.std = float(flow[train][mask[train]].std())

        self.x = torch.tensor(((filled - self.mean) / self.std).astype(np.float32), device=device)    # inputs (normalised)
        self.y = torch.tensor(np.nan_to_num(flow, nan=0.0).astype(np.float32), device=device)       # targets (original units)
        self.m = torch.tensor(mask.astype(np.float32), device=device)            # 1 = observed target
        self.device = device

        # ---- window start indices s: inputs s..s+7, targets s+8..s+11
        L = IN_LEN + OUT_LEN
        s = np.arange(self.T - L + 1)
        same = (self.split[s] == self.split[s + L - 1]) & (self.split[s] >= 0)
        self.starts = {}
        for k, name in enumerate(["train", "val", "test"]):
            idx = s[same & (self.split[s] == k)]
            if name == "train" and train_stride > 1:
                idx = idx[::train_stride]
            self.starts[name] = idx
        self.offsets = torch.arange(L, device=device)

    def batch(self, starts):
        """starts: 1-D LongTensor/ndarray of window starts -> x (B,8,N), y (B,4,N), mask (B,4,N)."""
        st = torch.as_tensor(starts, device=self.device, dtype=torch.long)
        idx = st[:, None] + self.offsets[None, :]                 # (B, 12)
        x = self.x[idx[:, :IN_LEN]]
        y = self.y[idx[:, IN_LEN:]]
        m = self.m[idx[:, IN_LEN:]]
        return x, y, m

    def target_times(self, name, h):
        """timestamps of horizon-h targets (h = 0..3) for every window of a split."""
        return self.ts[self.starts[name] + IN_LEN + h]

    def summary(self):
        out = dict(T=self.T, N=self.N, mean=self.mean, std=self.std, aggregation=self.agg)
        for k, v in self.starts.items():
            out[f"n_{k}_windows"] = int(len(v))
            if len(v):
                out[f"{k}_first_input"] = str(self.ts[v[0]])
                out[f"{k}_last_target"] = str(self.ts[v[-1] + IN_LEN + OUT_LEN - 1])
        return out
