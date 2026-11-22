"""Q6 - Experimental analysis and interpretation (no training; reads Q4/Q5 outputs).

(a) final comparison table + predicted-vs-actual plots for several sensors
(b) learned-graph analysis: overlap with the road graph, geographic distance, dependencies
    absent from the physical graph (checked against 2020 training-data correlations only)
(c) prints the numbers you need for the critical discussion

Run:  python q6_analysis.py [--k 10]
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import to_datetime_index, PROCESSED_FILE, META_FILE, RES_DIR, banner, load_json, plt, save_json, save_text, savefig

MODELS = [("temporal_only", "Temporal-only", "None"), ("stgnn_fixed", "STGNN", "Fixed"),
          ("stgnn_adaptive", "STGNN", "Adaptive"), ("stgnn_hybrid", "STGNN", "Hybrid")]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ============================================================================ (a)
def part_a():
    banner("Q6(a)  Forecasting comparison")
    rows, res = [], {}
    for key, model, adj in MODELS:
        r = load_json(f"metrics_{key}.json")
        if r is None:
            print(f"  [missing] {key} - run Q4/Q5 first"); continue
        res[key] = r
        rows.append({"Model": model, "Adjacency": adj, "MAE": r["test"]["MAE"], "RMSE": r["test"]["RMSE"],
                     "Training time": f"{r['train_time_s']/60:.1f} min ({r['epochs_run']} ep, {r['mean_epoch_time_s']:.1f} s/ep)",
                     "Val MAE": r["val"]["MAE"], "Val RMSE": r["val"]["RMSE"],
                     **{f"MAE@{15*(h+1)}min": r["test"]["MAE_h"][h] for h in range(4)}})
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(3)
    print(df[["Model", "Adjacency", "MAE", "RMSE", "Training time"]].to_string(index=False))
    df.to_csv(os.path.join(RES_DIR, "q6_final_table.csv"), index=False)
    try:
        md = df[["Model", "Adjacency", "MAE", "RMSE", "Training time"]].to_markdown(index=False)
        md_full = df.to_markdown(index=False)
    except Exception:
        md = df[["Model", "Adjacency", "MAE", "RMSE", "Training time"]].to_string(index=False); md_full = df.to_string(index=False)
    save_text("Test-set results (Apr-Dec 2021), original flow units\n\n" + md + "\n\nFull:\n\n" + md_full, "q6_final_table.md")

    # bar chart
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    labs = [f"{r['Model']}\n{r['Adjacency']}" for r in rows]
    ax[0].bar(labs, df["MAE"]); ax[0].set_title("Test MAE")
    ax[1].bar(labs, df["RMSE"], color="tab:orange"); ax[1].set_title("Test RMSE")
    for a, col in zip(ax, ["MAE", "RMSE"]):
        lo = df[col].min()
        a.set_ylim(lo * 0.95, df[col].max() * 1.02); a.grid(alpha=.3, axis="y")
        for i, v in enumerate(df[col]):
            a.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    savefig(fig, "q6a_metric_bars")

    # predicted vs actual
    P = {k: np.load(os.path.join(RES_DIR, f"preds_{k}.npz")) for k in res if os.path.exists(os.path.join(RES_DIR, f"preds_{k}.npz"))}
    if not P:
        return df, res
    base = next(iter(P.values()))
    sensors = base["sensors"]; ids = np.load(PROCESSED_FILE)["sensor_ids"]
    t_h1 = to_datetime_index(base["target_time_h1"])
    w0 = pd.Timestamp("2021-05-03")  # a Monday in the test period
    for h, hname in [(0, "15min"), (3, "60min")]:
        tt = t_h1 + pd.Timedelta(minutes=15 * h)
        sel = (tt >= w0) & (tt < w0 + pd.Timedelta(days=7))
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        for ax, si in zip(axes, range(3)):
            true = base["true"][sel, h, si].astype(float); msk = base["mask"][sel, h, si] > 0
            true[~msk] = np.nan
            ax.plot(tt[sel], true, "k", lw=1.6, label="actual")
            for k, lab in [(k, f"{m} {a}") for k, m, a in MODELS if k in P]:
                ax.plot(tt[sel], P[k]["pred"][sel, h, si], lw=.9, label=lab)
            ax.set_title(f"sensor {ids[sensors[si]]}  (horizon {hname})"); ax.grid(alpha=.3)
            ax.set_ylabel("flow")
        axes[0].legend(ncol=5, fontsize=8)
        fig.suptitle(f"Predicted vs actual traffic flow, test week of {w0.date()}, {hname} ahead", y=1.0)
        savefig(fig, f"q6a_pred_vs_actual_{hname}")
    # one-day zoom, all horizons, best model
    best = min(res, key=lambda k: res[k]["test"]["MAE"])
    day = (t_h1 >= w0 + pd.Timedelta(days=1)) & (t_h1 < w0 + pd.Timedelta(days=2))
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.6))
    for ax, si in zip(axes, range(3)):
        tr = base["true"][day, 0, si].astype(float); tr[base["mask"][day, 0, si] == 0] = np.nan
        ax.plot(t_h1[day], tr, "k", lw=1.6, label="actual")
        for h in range(4):
            ax.plot(t_h1[day] + pd.Timedelta(minutes=15 * h), P[best]["pred"][day, h, si], lw=.9, label=f"+{15*(h+1)} min")
        ax.set_title(f"{best}: sensor {ids[sensors[si]]}"); ax.tick_params(axis="x", rotation=30); ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    savefig(fig, "q6a_best_model_one_day_all_horizons")
    return df, res


# ============================================================================ (b)
def part_b(k=10):
    banner("Q6(b)  Graph structure analysis")
    path = os.path.join(RES_DIR, "adj_stgnn_adaptive.npy")
    if not os.path.exists(path):
        print("  adaptive adjacency not found - run Q5 first"); return {}
    Aa = np.load(path).astype(np.float64)
    d = np.load(PROCESSED_FILE)
    A = d["adj"]; flow = d["flow"]; ts = to_datetime_index(d["timestamps"])
    meta = pd.read_csv(META_FILE)
    N = A.shape[0]
    phys = (A != 0); np.fill_diagonal(phys, False)
    off = Aa.copy(); np.fill_diagonal(off, -np.inf)

    # top-k learned neighbours per node (excluding self)
    topk = np.argsort(-off, axis=1)[:, :k]
    rows = np.repeat(np.arange(N), k); cols = topk.ravel()
    in_phys = phys[rows, cols]
    density = phys.sum() / (N * (N - 1))
    # strongest edges globally
    flat = np.argsort(-off.ravel())[: 10 * N]
    gr, gc = np.unravel_index(flat, off.shape)
    in_phys_global = phys[gr, gc]
    print(f"  (1) top-{k} learned neighbours per node present in road graph: {in_phys.mean():.2%} "
          f"(random baseline = graph density {density:.2%}; x{in_phys.mean()/max(density,1e-12):.1f})")
    print(f"      strongest {10*N} learned edges overall: {in_phys_global.mean():.2%} also in road graph")

    # weight mass the learned graph puts on physical neighbours
    mass_phys = (Aa * phys).sum(1) / np.maximum(Aa.sum(1) - np.diag(Aa), 1e-12)
    print(f"      share of each row's off-diagonal learned weight on road-graph neighbours: mean {mass_phys.mean():.2%}")

    out = dict(k=k, topk_in_phys=float(in_phys.mean()), density=float(density),
               global_in_phys=float(in_phys_global.mean()), mass_on_phys=float(mass_phys.mean()))

    # (2) geography
    if {"Lat", "Lng"}.issubset(meta.columns):
        lat, lng = meta.Lat.to_numpy(), meta.Lng.to_numpy()
        D = haversine(lat[:, None], lng[:, None], lat[None, :], lng[None, :])
        rng = np.random.default_rng(0)
        rr, rc = rng.integers(0, N, 20000), rng.integers(0, N, 20000)
        keep = rr != rc
        d_learn, d_phys, d_rand = D[rows, cols], D[phys], D[rr[keep], rc[keep]]
        print(f"  (2) median straight-line distance: learned top-{k} {np.median(d_learn):.2f} km | "
              f"road-graph edges {np.median(d_phys):.2f} km | random pairs {np.median(d_rand):.2f} km")
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for v, lab in [(d_phys, "road-graph edges"), (d_learn, f"learned top-{k}"), (d_rand, "random pairs")]:
            ax.hist(v, bins=np.linspace(0, np.quantile(d_rand, .99), 60), density=True, histtype="step", lw=1.6, label=lab)
        ax.set_xlabel("haversine distance between sensors (km)"); ax.set_ylabel("density"); ax.legend(); ax.grid(alpha=.3)
        ax.set_title("Are learned connections geographically local?")
        savefig(fig, "q6b_distance_distribution")
        out.update(median_km_learned=float(np.median(d_learn)), median_km_phys=float(np.median(d_phys)),
                   median_km_random=float(np.median(d_rand)),
                   frac_learned_within_5km=float((d_learn < 5).mean()))

        # map of strongest learned edges coloured by physical membership
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(lng, lat, s=3, c="lightgrey", zorder=1)
        top_glob = np.argsort(-off.ravel())[:400]
        r4, c4 = np.unravel_index(top_glob, off.shape)
        for a_, b_ in zip(r4, c4):
            col = "tab:blue" if phys[a_, b_] else "tab:red"
            ax.plot([lng[a_], lng[b_]], [lat[a_], lat[b_]], c=col, lw=.6, alpha=.6, zorder=2)
        ax.plot([], [], c="tab:blue", label="also in road graph"); ax.plot([], [], c="tab:red", label="learned only")
        ax.legend(); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        ax.set_title("400 strongest learned connections")
        savefig(fig, "q6b_learned_edges_map")

    # (3) dependencies absent from the physical graph: are they real (correlated in TRAINING data)?
    tr = ts.year == 2020
    X = pd.DataFrame(flow[tr]).ffill().bfill().to_numpy()
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    C = (X.T @ X) / len(X)
    rng = np.random.default_rng(1)
    rr, rc = rng.integers(0, N, 20000), rng.integers(0, N, 20000); keep = rr != rc
    c_lonly = C[rows[~in_phys], cols[~in_phys]]
    c_lphys = C[rows[in_phys], cols[in_phys]]
    c_rand = C[rr[keep], rc[keep]]
    print(f"  (3) learned-only top-{k} edges: {(~in_phys).sum()} of {len(in_phys)} "
          f"({(~in_phys).mean():.1%}) are NOT in the road graph")
    print(f"      mean 2020 flow correlation: learned-only {np.mean(c_lonly):.3f} | learned & physical "
          f"{np.mean(c_lphys) if len(c_lphys) else float('nan'):.3f} | random pairs {np.mean(c_rand):.3f} | "
          f"road edges {C[phys].mean():.3f}")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(-0.2, 1, 60)
    ax.hist(c_rand, bins=bins, density=True, histtype="step", lw=1.6, label="random pairs")
    ax.hist(C[phys], bins=bins, density=True, histtype="step", lw=1.6, label="road-graph edges")
    ax.hist(c_lonly, bins=bins, density=True, histtype="step", lw=1.6, label="learned-only edges")
    ax.set_xlabel("Pearson correlation of 2020 flow"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("Do learned-only connections link similarly behaving sensors?")
    savefig(fig, "q6b_correlation_of_learned_edges")
    out.update(n_learned_only=int((~in_phys).sum()), corr_learned_only=float(np.mean(c_lonly)),
               corr_learned_phys=float(np.mean(c_lphys)) if len(c_lphys) else None,
               corr_random=float(np.mean(c_rand)), corr_road=float(C[phys].mean()))

    # in-degree concentration of the learned graph ("hub" sensors)
    indeg = Aa.sum(0)
    out["learned_in_weight_top5_share"] = float(np.sort(indeg)[-5:].sum() / indeg.sum())
    print(f"      top-5 sensors receive {out['learned_in_weight_top5_share']:.1%} of all learned edge weight "
          f"(uniform = {5/N:.1%}) -> hub structure" if out['learned_in_weight_top5_share'] > 3 * 5 / N else "")

    # side-by-side: road graph vs learned graph on the same sensor ordering (sorted by freeway)
    order = np.lexsort((meta.Lat.to_numpy(), meta.Fwy.astype(str).to_numpy())) if "Fwy" in meta else np.arange(N)
    An, _ = __import__("data_utils").normalize_adj(A, "row")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
    for a_, M, t in [(ax[0], An, "road graph D^-1(A+I)"), (ax[1], Aa, "learned adaptive graph")]:
        S = M[np.ix_(order, order)].copy(); np.fill_diagonal(S, 0)
        im = a_.imshow(S, cmap="magma", vmin=0, vmax=np.quantile(S[S > 0], .99), interpolation="nearest")
        a_.set_title(t + "\n(sensors sorted by freeway, then latitude; diagonal removed)"); fig.colorbar(im, ax=a_, fraction=.046)
    savefig(fig, "q6b_road_vs_learned_sorted")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=10, help="top-k learned neighbours used in the analysis")
    a = p.parse_args()
    df, res = part_a()
    gb = part_b(a.k)
    banner("Q6(c)  Numbers for the critical discussion")
    if res:
        for key, m, adj in MODELS:
            if key in res:
                print(f"  {m:13s} {adj:8s}: test MAE {res[key]['test']['MAE']:.3f}  RMSE {res[key]['test']['RMSE']:.3f}")
        h = res.get("stgnn_hybrid", {})
        if "alpha" in h:
            print(f"  learned hybrid alpha = {h['alpha']:.3f} (weight on the road graph)")
    if gb:
        print(f"  learned top-{gb['k']} edges in road graph: {gb['topk_in_phys']:.1%}; learned-only edges have mean "
              f"training correlation {gb['corr_learned_only']:.3f} vs random {gb['corr_random']:.3f}")
    save_json(dict(table=df.to_dict(orient="records") if len(df) else [], graph=gb), "q6_summary.json")
    print("\nDone with Q6.")
