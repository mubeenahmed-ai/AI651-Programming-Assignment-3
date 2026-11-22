"""Download + memory-safe reading of the LargeST raw files.

Kaggle dataset liuxu77/largest contains (California-wide):
    ca_meta.csv           sensor metadata (ID, Lat, Lng, District, County, Fwy, Lanes, Type, Direction, ID2)
    ca_rn_adj.npy         8600 x 8600 road-network (distance-based) adjacency
    ca_his_raw_YYYY.h5    5-minute traffic flow, rows = timestamps, columns = sensor IDs

San Diego = District 11 (716 sensors), exactly as in the LargeST repo's SD generation script.
The CA-wide h5 files are large, so we read only the SD columns, in row chunks, with h5py.
"""
import os
import subprocess
import zipfile

import numpy as np
import pandas as pd

KAGGLE_DATASET = "liuxu77/largest"
META_NAME = "ca_meta.csv"
ADJ_NAME = "ca_rn_adj.npy"
HIS_NAME = "ca_his_raw_{year}.h5"
SD_DISTRICT = 11


# ----------------------------------------------------------------------------- download
def _kaggle_download_file(fname, raw_dir):
    print(f"[kaggle] downloading {fname} ...")
    cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-f", fname, "-p", raw_dir, "--force"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-500:], r.stderr[-500:])
    if r.returncode != 0:
        raise RuntimeError(f"kaggle download of {fname} failed (see message above).\n"
                           "Check ~/.kaggle/kaggle.json and that you accepted the dataset on kaggle.com.")
    # Kaggle may deliver the file zipped
    for cand in (os.path.join(raw_dir, fname + ".zip"), os.path.join(raw_dir, os.path.splitext(fname)[0] + ".zip")):
        if os.path.exists(cand):
            with zipfile.ZipFile(cand) as z:
                z.extractall(raw_dir)
            os.remove(cand)
    if not os.path.exists(os.path.join(raw_dir, fname)):
        raise FileNotFoundError(f"{fname} not found after download. Files listed by Kaggle:\n" + list_kaggle_files())


def list_kaggle_files():
    r = subprocess.run(["kaggle", "datasets", "files", KAGGLE_DATASET], capture_output=True, text=True)
    return r.stdout + r.stderr


def ensure_raw_files(raw_dir, years=(2020, 2021)):
    needed = [META_NAME, ADJ_NAME] + [HIS_NAME.format(year=y) for y in years]
    for f in needed:
        p = os.path.join(raw_dir, f)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            print(f"[raw] found {p}  ({os.path.getsize(p)/1e9:.2f} GB)")
        else:
            _kaggle_download_file(f, raw_dir)
    return {f: os.path.join(raw_dir, f) for f in needed}


# ----------------------------------------------------------------------------- metadata / graph
def load_sd_meta_and_adj(raw_dir):
    meta = pd.read_csv(os.path.join(raw_dir, META_NAME))
    sd = meta[meta["District"] == SD_DISTRICT].copy()
    # Row index of each SD sensor inside the CA adjacency. LargeST stores it in ID2;
    # if that column is absent, the CA adjacency follows the ca_meta row order.
    if "ID2" in sd.columns:
        adj_idx = sd["ID2"].to_numpy().astype(int)
        src = "ID2 column"
    else:
        adj_idx = sd.index.to_numpy()
        src = "ca_meta row position"
    sd = sd.reset_index(drop=True)
    sd["adj_idx"] = adj_idx
    adj_ca = np.load(os.path.join(raw_dir, ADJ_NAME), mmap_mode="r")
    adj = np.asarray(adj_ca[np.ix_(adj_idx, adj_idx)], dtype=np.float32)
    print(f"[meta] {len(sd)} San Diego sensors (District {SD_DISTRICT}); adjacency rows taken via {src}; "
          f"CA adj shape {adj_ca.shape}")
    return sd, adj


# ----------------------------------------------------------------------------- HDF5 reading
def _decode(arr):
    out = []
    for v in arr:
        if isinstance(v, bytes):
            v = v.decode()
        out.append(str(v))
    return np.array(out)


def read_h5_columns(path, wanted_ids, chunk_rows=6000):
    """Return (timestamps DatetimeIndex, values float32 [T, len(wanted_ids)]) in wanted_ids order."""
    wanted_ids = [str(w) for w in wanted_ids]
    try:
        try:
            import hdf5plugin  # noqa: F401  (enables blosc etc. if the file is compressed)
        except ImportError:
            pass
        import h5py
        with h5py.File(path, "r") as f:
            key = "df" if "df" in f else list(f.keys())[0]
            g = f[key]
            nblocks = len([k for k in g.keys() if k.startswith("block") and k.endswith("_values")])
            if nblocks != 1 or "axis0" not in g or "axis1" not in g:
                raise ValueError("unexpected pandas layout")
            cols = _decode(g["axis0"][:])
            idx = g["axis1"][:]
            vals = g["block0_values"]
            pos = {c: i for i, c in enumerate(cols)}
            missing = [w for w in wanted_ids if w not in pos]
            if missing:
                raise KeyError(f"{len(missing)} SD sensor ids not in {os.path.basename(path)}, e.g. {missing[:5]}")
            sel = np.array([pos[w] for w in wanted_ids])
            T = len(idx)
            if vals.shape == (T, len(cols)):
                time_major = True
            elif vals.shape == (len(cols), T):
                time_major = False
            else:
                raise ValueError(f"block shape {vals.shape} does not match axes ({T}, {len(cols)})")
            out = np.empty((T, len(sel)), dtype=np.float32)
            order = np.argsort(sel)
            sorted_sel = sel[order]
            inv = np.argsort(order)
            if time_major:
                for s in range(0, T, chunk_rows):
                    block = vals[s:s + chunk_rows, :]          # contiguous row chunk
                    out[s:s + chunk_rows] = block[:, sorted_sel][:, inv]
            else:
                out[:] = vals[sorted_sel, :][inv].T
            from common import to_datetime_index
            ts = to_datetime_index(idx) if np.issubdtype(idx.dtype, np.integer) else pd.to_datetime(_decode(idx))
        print(f"[h5py] {os.path.basename(path)}: {T} rows, {len(sel)} SD columns")
        return pd.DatetimeIndex(ts), out
    except Exception as e:  # pragma: no cover - fallback path
        print(f"[h5py] fast path failed ({e}); falling back to pandas.read_hdf (needs more RAM)")
        df = pd.read_hdf(path)
        df.columns = [str(c) for c in df.columns]
        df = df[wanted_ids]
        return pd.DatetimeIndex(df.index), df.to_numpy(np.float32)
