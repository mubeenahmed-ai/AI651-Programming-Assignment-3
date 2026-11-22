"""Q3 - Dataset preparation and graph construction (LargeST, San Diego subset).

(a) extract the 716 SD sensors for 2020-2021, resample 5 min -> 15 min, missing-value analysis,
    one-week plot of three sensors
(b) SD road-distance adjacency: #nodes, #edges, symmetry, heatmap, degree distribution
(c) one-hop / two-hop neighbourhood of one sensor (+ correlation evidence from training data)

Run:  python q3_dataset.py [--agg sum|mean] [--force]
Outputs: PA3_DATA_DIR/processed/sd_flow15.npz (+ sd_meta.csv), figures, results/q3_summary.json
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import (META_FILE, to_datetime_index, datetime_to_ns, PROCESSED_FILE, RAW_DIR, banner, plt, save_json, savefig, set_seed)
from largest_io import ensure_raw_files, load_sd_meta_and_adj, read_h5_columns, HIS_NAME

set_seed()
YEARS = (2020, 2021)


# ============================================================================ (a)
def resample_15min(ts, vals, how):
    """5-min -> 15-min.  PeMS 'flow' = vehicles counted in each 5-min interval, so the
    physically consistent aggregate is the SUM (vehicles per 15 min).  With missing 5-min
    readings we use 3 * mean(available) (= the sum when all 3 exist; a rate-preserving
    imputation when 1-2 exist) and NaN when all 3 are missing.  how='mean' gives the
    average 5-min count instead (LargeST benchmark convention)."""
    df = pd.DataFrame(vals, index=ts)
    df = df[~df.index.duplicated()].sort_index()
    full = pd.date_range(df.index.min().floor("15min"), df.index.max(), freq="5min")
    df = df.reindex(full)  # make gaps explicit
    r = df.resample("15min", label="left", closed="left")
    mean = r.mean()
    cnt = r.count()
    out = mean * 3.0 if how == "sum" else mean
    out[cnt == 0] = np.nan
    return out.index, out.to_numpy(np.float32), cnt.to_numpy(np.int8), df.isna().to_numpy().mean()


def part_a(args):
    banner("Q3(a)  Data extraction and resampling")
    ensure_raw_files(RAW_DIR, YEARS)
    meta, adj = load_sd_meta_and_adj(RAW_DIR)
    ids = meta["ID"].astype(str).tolist()
    assert len(ids) == 716, f"expected 716 SD sensors, got {len(ids)}"

    all_ts, all_vals, all_cnt, raw_missing = [], [], [], {}
    for y in YEARS:
        ts, v = read_h5_columns(os.path.join(RAW_DIR, HIS_NAME.format(year=y)), ids)
        keep = ts.year == y
        ts, v = ts[keep], v[keep]
        print(f"  {y}: {len(ts)} 5-min rows, {ts.min()} -> {ts.max()}, raw NaN rate {np.isnan(v).mean():.3%}")
        t15, v15, c15, miss5 = resample_15min(ts, v, args.agg)
        raw_missing[y] = float(miss5)
        all_ts.append(t15); all_vals.append(v15); all_cnt.append(c15)
    ts15 = all_ts[0].append(all_ts[1])
    flow = np.concatenate(all_vals)
    cnt = np.concatenate(all_cnt)
    # enforce a perfectly regular 15-min grid over both years
    grid = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31 23:45", freq="15min")
    df = pd.DataFrame(flow, index=ts15).reindex(grid)
    flow = df.to_numpy(np.float32)
    cnt = pd.DataFrame(cnt, index=ts15).reindex(grid).fillna(0).to_numpy(np.int8)
    ts15 = grid
    print(f"  15-min tensor: X in R^(T x N x F) = {flow.shape[0]} x {flow.shape[1]} x 1")

    # ---------------- missing values
    miss = np.isnan(flow)
    neg = np.nansum(flow < 0)
    partial = ((cnt > 0) & (cnt < 3)).mean()
    per_sensor = miss.mean(0)
    print(f"  missing 15-min values: {miss.mean():.3%}  (5-min raw: " +
          ", ".join(f"{y}: {m:.3%}" for y, m in raw_missing.items()) + ")")
    print(f"  15-min slots built from only 1-2 of 3 readings: {partial:.3%}; negative values: {neg}")
    print(f"  sensors with >50% missing: {(per_sensor > .5).sum()}, fully missing: {(per_sensor == 1).sum()}")
    print("  Handling: (1) missing slots stay NaN in the stored tensor and are EXCLUDED from the loss and\n"
          "  from all MAE/RMSE computations (masked metrics). (2) For model INPUTS only, gaps are filled\n"
          "  causally by forward-fill (last observed value, never uses the future); values still missing\n"
          "  at the start are filled with that sensor's 2020 (training) mean. No val/test statistics are used.")

    np.savez_compressed(PROCESSED_FILE, flow=flow, timestamps=datetime_to_ns(ts15),
                        sensor_ids=np.array(ids), adj=adj, count5=cnt, agg=np.array(args.agg))
    meta.to_csv(META_FILE, index=False)
    print(f"  saved {PROCESSED_FILE}")

    # missing-value figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    ax[0].hist(per_sensor * 100, bins=50)
    ax[0].set_xlabel("missing rate per sensor (%)"); ax[0].set_ylabel("#sensors"); ax[0].set_title("Missing rate per sensor")
    daily = pd.Series(miss.mean(1), index=ts15).resample("D").mean() * 100
    ax[1].plot(daily.index, daily.values)
    ax[1].set_ylabel("% missing (daily)"); ax[1].set_title("Missing rate over time (all sensors)")
    savefig(fig, "q3a_missing_values")

    # one-week plot: three sensors with good coverage, from different freeways if possible
    week = (ts15 >= "2020-02-03") & (ts15 < "2020-02-10")  # Mon-Sun, pre-COVID
    cov = (~miss[week]).mean(0)
    order = np.argsort(-cov)
    chosen, fwys = [], set()
    for j in order:
        f = meta.loc[j, "Fwy"] if "Fwy" in meta else j
        if f not in fwys or len(order) < 3:
            chosen.append(int(j)); fwys.add(f)
        if len(chosen) == 3:
            break
    unit = "vehicles / 15 min" if args.agg == "sum" else "vehicles / 5 min (avg)"
    fig, ax = plt.subplots(figsize=(12, 4))
    for j in chosen:
        lab = f"sensor {ids[j]}" + (f" ({meta.loc[j, 'Fwy']})" if "Fwy" in meta else "")
        ax.plot(ts15[week], flow[week, j], lw=1, label=lab)
    ax.set_ylabel(f"flow [{unit}]"); ax.set_title("Traffic flow of three SD sensors, 3-9 Feb 2020 (15-min)")
    ax.legend(); ax.grid(alpha=.3)
    savefig(fig, "q3a_three_sensors_one_week")
    return dict(T=int(flow.shape[0]), N=int(flow.shape[1]), aggregation=args.agg, unit=unit,
                missing_15min=float(miss.mean()), missing_5min_raw=raw_missing, partial_slots=float(partial),
                sensors_gt50pct_missing=int((per_sensor > .5).sum()), plotted_sensors=[ids[j] for j in chosen])


# ============================================================================ (b)
def part_b():
    banner("Q3(b)  Constructing the traffic graph")
    d = np.load(PROCESSED_FILE)
    A = d["adj"].astype(np.float64)
    ids = d["sensor_ids"]
    meta = pd.read_csv(META_FILE)
    assert (meta["ID"].astype(str).to_numpy() == ids).all(), "adjacency / tensor order mismatch"
    print("  Ordering check passed: adjacency rows/cols and tensor columns both follow sd_meta order.")
    N = A.shape[0]
    diag = np.diag(A)
    off = A.copy(); np.fill_diagonal(off, 0)
    nnz_all, nnz_off = int((A != 0).sum()), int((off != 0).sum())
    sym = bool(np.allclose(A, A.T))
    pattern_sym = bool(((A != 0) == (A.T != 0)).all())
    asym_pairs = int(((off != 0) & (off.T == 0)).sum())
    print(f"  nodes: {N}")
    print(f"  nonzero entries: {nnz_all} (incl. {int((diag != 0).sum())} diagonal); directed off-diagonal edges: {nnz_off}")
    print(f"  density (off-diag): {nnz_off / (N * (N - 1)):.4%};  weight range: [{A[A>0].min():.4f}, {A.max():.4f}]")
    print(f"  symmetric (values): {sym};  symmetric (sparsity pattern): {pattern_sym};  one-directional edges: {asym_pairs}")
    if not sym:
        print("  -> The graph is DIRECTED: road distance from i to j differs from j to i (one-way\n"
              "     carriageways / different driving routes). We therefore use row (random-walk)\n"
              "     normalisation D^-1 (A + I) for the models instead of the symmetric D^-1/2 A D^-1/2.")

    out_deg = (off != 0).sum(1); in_deg = (off != 0).sum(0)
    print(f"  out-degree: mean {out_deg.mean():.2f}, median {np.median(out_deg):.0f}, min {out_deg.min()}, "
          f"max {out_deg.max()}, isolated nodes {(out_deg == 0).sum()}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    Am = np.ma.masked_where(A == 0, A)
    im = ax[0].imshow(Am, cmap="viridis", interpolation="nearest")
    ax[0].set_title(f"SD adjacency ({N}x{N}), nonzero weights"); ax[0].set_xlabel("sensor j"); ax[0].set_ylabel("sensor i")
    fig.colorbar(im, ax=ax[0], fraction=.046)
    im = ax[1].imshow(Am[:100, :100], cmap="viridis", interpolation="nearest")
    ax[1].set_title("zoom: first 100 sensors"); fig.colorbar(im, ax=ax[1], fraction=.046)
    savefig(fig, "q3b_adjacency_heatmap")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.arange(0, max(out_deg.max(), in_deg.max()) + 2) - .5
    ax.hist(out_deg, bins=bins, alpha=.7, label="out-degree (row nnz)")
    if not sym:
        ax.hist(in_deg, bins=bins, alpha=.5, label="in-degree (col nnz)")
    ax.set_xlabel("number of neighbours (excluding self)"); ax.set_ylabel("#sensors"); ax.legend()
    ax.set_title("Neighbour-count distribution"); ax.grid(alpha=.3)
    savefig(fig, "q3b_degree_distribution")
    print("  Meaning of A_ij != 0: sensor j is within the road-network distance threshold of sensor i;\n"
          "  the value is a Gaussian-kernel weight exp(-d_ij^2 / sigma^2) of the road distance d_ij, so\n"
          "  larger = closer along the road network = expected stronger traffic interaction.")
    return dict(nodes=N, nonzero_entries=nnz_all, offdiag_edges=nnz_off, diag_nonzero=int((diag != 0).sum()),
                symmetric=sym, pattern_symmetric=pattern_sym, one_directional_edges=asym_pairs,
                out_degree_mean=float(out_deg.mean()), out_degree_median=float(np.median(out_deg)),
                out_degree_max=int(out_deg.max()), isolated=int((out_deg == 0).sum()))


# ============================================================================ (c)
def part_c(sensor=None):
    banner("Q3(c)  Graph neighbourhoods")
    d = np.load(PROCESSED_FILE)
    A = d["adj"]; flow = d["flow"]; ts = to_datetime_index(d["timestamps"]); ids = d["sensor_ids"]
    meta = pd.read_csv(META_FILE)
    N = A.shape[0]
    B = (A != 0).astype(np.int64); np.fill_diagonal(B, 0)
    deg = B.sum(1)
    i = int(np.argmin(np.abs(deg - np.median(deg[deg > 0])))) if sensor is None else int(sensor)
    one = set(np.nonzero(B[i])[0].tolist())
    reach2 = (B + np.eye(N, dtype=np.int64)) @ (B + np.eye(N, dtype=np.int64))
    two = set(np.nonzero(reach2[i])[0].tolist()) - {i}
    two_only = two - one
    print(f"  selected sensor index {i} (ID {ids[i]}), out-degree {deg[i]}")
    print(f"  one-hop neighbours ({len(one)}): {[str(ids[j]) for j in sorted(one)]}")
    print(f"  two-hop neighbourhood (<=2 edges, {len(two)} nodes, {len(two_only)} new at exactly 2 hops): "
          f"{[str(ids[j]) for j in sorted(two_only)]}")

    # evidence: correlation with sensor i on TRAINING data (2020) only
    tr = ts.year == 2020
    X = pd.DataFrame(flow[tr]).ffill().bfill().to_numpy()
    xi = X[:, i]
    corr = np.array([np.corrcoef(xi, X[:, j])[0, 1] if X[:, j].std() > 0 else np.nan for j in range(N)])
    rest = list(set(range(N)) - two - {i})
    stats = {k: float(np.nanmean(corr[list(v)])) if len(v) else float("nan")
             for k, v in dict(one_hop=one, two_hop_only=two_only, others=rest).items()}
    print(f"  mean Pearson corr with sensor (2020): 1-hop {stats['one_hop']:.3f}, exactly-2-hop "
          f"{stats['two_hop_only']:.3f}, rest of network {stats['others']:.3f}")

    if {"Lat", "Lng"}.issubset(meta.columns):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(meta.Lng, meta.Lat, s=4, c="lightgrey", label="other sensors")
        if two_only:
            ax.scatter(meta.Lng[list(two_only)], meta.Lat[list(two_only)], s=18, c="tab:orange", label="2-hop")
        if one:
            ax.scatter(meta.Lng[list(one)], meta.Lat[list(one)], s=26, c="tab:blue", label="1-hop")
        ax.scatter(meta.Lng[i], meta.Lat[i], s=120, c="red", marker="*", label=f"sensor {ids[i]}")
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend()
        ax.set_title("One-hop and two-hop neighbourhood")
        savefig(fig, "q3c_neighbourhood_map")
        # zoom
        sel = list(two | {i})
        fig, ax = plt.subplots(figsize=(6, 6))
        for j in one:
            ax.plot([meta.Lng[i], meta.Lng[j]], [meta.Lat[i], meta.Lat[j]], c="tab:blue", lw=.8)
        ax.scatter(meta.Lng[sel], meta.Lat[sel], s=14, c=["red" if j == i else "tab:blue" if j in one else "tab:orange" for j in sel])
        ax.set_title("Zoom on the 2-hop neighbourhood (lines = 1-hop edges)")
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        savefig(fig, "q3c_neighbourhood_zoom")

    # how fast does the receptive field grow with depth? (over-smoothing discussion)
    Bi = (B + np.eye(N, dtype=np.int64)) > 0
    R = np.eye(N, dtype=bool)
    sizes = []
    for k in range(1, 11):
        R = (R.astype(np.int64) @ Bi.astype(np.int64)) > 0
        sizes.append(float(R.sum(1).mean()))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(range(1, 11), sizes, "o-")
    ax.set_xlabel("number of graph-conv layers (hops)"); ax.set_ylabel("mean receptive field (#nodes)")
    ax.set_title("Receptive-field growth with depth"); ax.grid(alpha=.3)
    savefig(fig, "q3c_receptive_field_growth")
    print("  mean receptive-field size for 1..10 hops: " + ", ".join(f"{s:.0f}" for s in sizes))
    return dict(sensor_index=i, sensor_id=str(ids[i]), one_hop=[str(ids[j]) for j in sorted(one)],
                two_hop_only=[str(ids[j]) for j in sorted(two_only)], corr=stats, receptive_field=sizes)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agg", choices=["sum", "mean"], default="sum",
                   help="5->15 min aggregation (sum for interval vehicle counts)")
    p.add_argument("--force", action="store_true", help="rebuild processed file even if cached")
    p.add_argument("--sensor", type=int, default=None, help="sensor index for Q3(c)")
    args = p.parse_args()
    summary = {}
    if args.force or not os.path.exists(PROCESSED_FILE):
        summary["a"] = part_a(args)
    else:
        print(f"[cache] using {PROCESSED_FILE} (use --force to rebuild); Q3(a) figures from the first run are kept.")
        prev = __import__("common").load_json("q3_summary.json", {}) or {}
        summary["a"] = prev.get("a", {})
    summary["b"] = part_b()
    summary["c"] = part_c(args.sensor)
    save_json(summary, "q3_summary.json")
    print("\nDone with Q3.")
