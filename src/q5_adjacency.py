"""Q5 - Fixed versus learnable adjacency (same encoder, head, splits, protocol for all).

(a) fixed    : road-distance graph, normalised (row-normalised D^-1(A+I) if A is directed,
               symmetric D^-1/2(A+I)D^-1/2 otherwise); constant during training.
               -> identical to the Q4(b) run, reused if already trained.
(b) adaptive : A = Softmax_row(ReLU(E1 E2^T)), E1,E2 in R^{N x d_e}, random init, learned.
               --topk k keeps the k strongest entries per row BEFORE the softmax.
(c) hybrid   : A = alpha * A_fixed_row + (1-alpha) * A_adaptive, alpha learned (or --alpha).

Run:  python q5_adjacency.py [--emb_dim 10 --topk 0 --alpha <float>] [same training flags as Q4]
"""
import os

import numpy as np
import pandas as pd

from common import RES_DIR, banner, plt, save_json, save_text, savefig, set_seed
from data_utils import TrafficData, normalize_adj
from train import get_args, plot_sensors, run_experiment


def show_adjs(mats, titles, name, zoom=100):
    fig, ax = plt.subplots(2, len(mats), figsize=(5.2 * len(mats), 9.5))
    for j, (M, t) in enumerate(zip(mats, titles)):
        for r, (S, sub) in enumerate([(M, "full"), (M[:zoom, :zoom], f"first {zoom} sensors")]):
            vmax = np.quantile(S[S > 0], 0.99) if (S > 0).any() else 1
            im = ax[r, j].imshow(S, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
            ax[r, j].set_title(f"{t} ({sub})"); fig.colorbar(im, ax=ax[r, j], fraction=.046)
    fig.suptitle("Normalised adjacency matrices (colour clipped at the 99th percentile of nonzeros)")
    return savefig(fig, name)


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    data = TrafficData(args.device, args.train_stride)
    keep = plot_sensors(data)

    banner("Q5(a)  Fixed adjacency")
    A_auto, mode = normalize_adj(data.adj, "auto")
    print(f"  normalisation used: {mode}  ({'D^-1/2 (A+I) D^-1/2' if mode == 'sym' else 'D^-1 (A+I), rows sum to 1 (directed graph)'})")
    fixed = run_experiment("stgnn_fixed", "fixed", args, data, keep)

    banner("Q5(b)  Learnable (adaptive) adjacency")
    print(f"  E1, E2 in R^({data.N} x {args.emb_dim}); A = softmax_row(relu(E1 E2^T)); "
          f"top-k = {args.topk if args.topk else 'none (dense)'}")
    adaptive = run_experiment("stgnn_adaptive", "adaptive", args, data, keep)

    banner("Q5(c)  Hybrid adjacency")
    print("  A_hybrid = alpha * D^-1(A_fixed+I) + (1-alpha) * A_adaptive  (both row-stochastic)"
          + (f", alpha fixed = {args.alpha}" if args.alpha is not None else ", alpha learned (init 0.5)"))
    hybrid = run_experiment("stgnn_hybrid", "hybrid", args, data, keep)
    if "alpha" in hybrid:
        print(f"  final alpha = {hybrid['alpha']:.4f}")

    rows = []
    for r, lab in [(fixed, "Fixed"), (adaptive, "Adaptive"), (hybrid, "Hybrid")]:
        rows.append({"Adjacency": lab, "Val MAE": r["val"]["MAE"], "Val RMSE": r["val"]["RMSE"],
                     "Test MAE": r["test"]["MAE"], "Test RMSE": r["test"]["RMSE"],
                     "Params": r["params"], "Epochs": r["epochs_run"],
                     "Train time (min)": r["train_time_s"] / 60, "s/epoch": r["mean_epoch_time_s"]})
    df = pd.DataFrame(rows).round(3)
    print("\n" + df.to_string(index=False))
    df.to_csv(os.path.join(RES_DIR, "q5_comparison.csv"), index=False)
    try:
        save_text(df.to_markdown(index=False), "q5_comparison.md")
    except Exception:
        save_text(df.to_string(index=False), "q5_comparison.md")

    # visualise the three graphs
    mats = [np.load(os.path.join(RES_DIR, f"adj_{n}.npy")) for n in ("stgnn_fixed", "stgnn_adaptive", "stgnn_hybrid")]
    show_adjs(mats, ["Fixed (road distance)", "Learned (adaptive)", "Hybrid"], "q5_adjacency_matrices")

    # row statistics of the learned graph: how concentrated is each node's attention?
    Aa = mats[1]
    ent = -(Aa * np.log(Aa + 1e-12)).sum(1)
    top1 = Aa.max(1)
    nz_eff = np.exp(ent)  # effective number of neighbours
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].hist(nz_eff, bins=50); ax[0].set_xlabel("effective #neighbours  exp(row entropy)")
    ax[0].set_title("Learned graph: neighbourhood size per node")
    ax[1].hist(top1, bins=50); ax[1].set_xlabel("largest weight in row"); ax[1].set_title("Learned graph: strongest edge per node")
    for a in ax:
        a.grid(alpha=.3)
    savefig(fig, "q5b_learned_graph_rows")

    # learning curves + alpha
    fig, ax = plt.subplots(1, 2 if "alpha" in hybrid else 1, figsize=(11, 3.4))
    ax = np.atleast_1d(ax)
    for r, lab in [(fixed, "Fixed"), (adaptive, "Adaptive"), (hybrid, "Hybrid")]:
        ax[0].plot([h["val_mae"] for h in r["history"]], label=lab)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("val MAE"); ax[0].set_title("Validation MAE"); ax[0].legend(); ax[0].grid(alpha=.3)
    if "alpha" in hybrid:
        ax[1].plot([h["alpha"] for h in hybrid["history"]], "o-")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("alpha"); ax[1].set_title("Hybrid: learned weight on fixed graph"); ax[1].grid(alpha=.3)
    savefig(fig, "q5_learning_curves")
    save_json(dict(comparison=rows, alpha=hybrid.get("alpha"), fixed_normalisation=mode,
                   adaptive_eff_neighbours_mean=float(nz_eff.mean()),
                   adaptive_top1_mean=float(top1.mean())), "q5_summary.json")
    print("\nDone with Q5.")
