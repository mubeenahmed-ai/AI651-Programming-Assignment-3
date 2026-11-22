"""Q4 - Spatio-temporal traffic forecasting.

(a) forecasting windows + chronological splits + train-only normalisation
(b) STGNN = temporal CNN encoder -> graph conv (fixed road graph) -> MLP head
(c) temporal-only baseline (same encoder + head, no graph conv) and comparison

The fixed-graph STGNN trained here is the same run reported in Q5(a) (identical protocol),
so Q5 reuses it instead of training it twice.

Run:  python q4_stgnn.py [--epochs 30 --hidden 32 --batch_size 64 ...]
"""
import numpy as np
import pandas as pd

from common import banner, plt, save_json, savefig, save_text, set_seed
from data_utils import TrafficData
from train import get_args, plot_sensors, run_experiment


def part_a(data):
    banner("Q4(a)  Forecasting windows")
    s = data.summary()
    print(f"  X_t = [x_(t-7) .. x_t] -> {8} x {data.N} x 1,   Y_t = [x_(t+1) .. x_(t+4)] -> 4 x {data.N} x 1")
    for k in ("train", "val", "test"):
        print(f"  {k:5s}: {s[f'n_{k}_windows']:6d} windows | first input {s.get(f'{k}_first_input')} | "
              f"last target {s.get(f'{k}_last_target')}")
    print(f"  normalisation (2020 observed values only): mean = {data.mean:.3f}, std = {data.std:.3f}")
    print("  A window is kept only if its 12 timestamps lie inside one split -> no window crosses a boundary.")
    # timeline figure
    daily = pd.Series(((data.y * data.m).sum(1) / data.m.sum(1).clamp(min=1)).cpu().numpy(), index=data.ts).resample("D").mean()
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(daily.index, daily.values, c="k", lw=.8)
    for (a, b, c, lab) in [("2020-01-01", "2021-01-01", "tab:blue", "train (2020)"),
                           ("2021-01-01", "2021-04-01", "tab:orange", "val (Jan-Mar 2021)"),
                           ("2021-04-01", "2022-01-01", "tab:green", "test (Apr-Dec 2021)")]:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=c, alpha=.15, label=lab)
    ax.set_ylabel("network-mean flow (daily)"); ax.legend(loc="lower right"); ax.set_title("Chronological splits")
    savefig(fig, "q4a_splits")
    return s


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    data = TrafficData(args.device, args.train_stride)
    sa = part_a(data)
    keep = plot_sensors(data)

    banner("Q4(b)  STGNN with the fixed road-distance graph")
    stgnn = run_experiment("stgnn_fixed", "fixed", args, data, keep)

    banner("Q4(c)  Temporal-only baseline")
    temp = run_experiment("temporal_only", "none", args, data, keep)

    rows = []
    for r, lab in [(temp, "Temporal-only"), (stgnn, "STGNN (fixed graph)")]:
        rows.append({"Model": lab, "Val MAE": r["val"]["MAE"], "Val RMSE": r["val"]["RMSE"],
                     "Test MAE": r["test"]["MAE"], "Test RMSE": r["test"]["RMSE"],
                     "Params": r["params"], "Train time (min)": r["train_time_s"] / 60})
    df = pd.DataFrame(rows).round(3)
    print("\n" + df.to_string(index=False))
    gain = 100 * (temp["test"]["MAE"] - stgnn["test"]["MAE"]) / temp["test"]["MAE"]
    print(f"\nRelative test-MAE change from adding the graph conv: {gain:+.2f}% "
          f"({'improvement' if gain > 0 else 'no improvement'})")
    df.to_csv(__import__("os").path.join(__import__("common").RES_DIR, "q4_comparison.csv"), index=False)
    try:
        save_text(df.to_markdown(index=False), "q4_comparison.md")
    except Exception:
        save_text(df.to_string(index=False), "q4_comparison.md")

    # per-horizon + learning curves
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    hz = [15, 30, 45, 60]
    for r, lab in [(temp, "Temporal-only"), (stgnn, "STGNN fixed")]:
        ax[0].plot(hz, r["test"]["MAE_h"], "o-", label=lab)
        ax[1].plot(hz, r["test"]["RMSE_h"], "o-", label=lab)
        ax[2].plot([h["val_mae"] for h in r["history"]], label=f"{lab} (val)")
        ax[2].plot([h["train_mae"] for h in r["history"]], "--", label=f"{lab} (train)")
    ax[0].set_title("Test MAE per horizon"); ax[1].set_title("Test RMSE per horizon"); ax[2].set_title("Learning curves (MAE)")
    for a in ax[:2]:
        a.set_xlabel("horizon (min)")
    ax[2].set_xlabel("epoch")
    for a in ax:
        a.grid(alpha=.3); a.legend()
    savefig(fig, "q4c_temporal_vs_graph")
    save_json(dict(windows=sa, comparison=rows, test_mae_gain_pct=gain), "q4_summary.json")
    print("\nDone with Q4.")
