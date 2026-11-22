"""Shared training / evaluation for Q4-Q5 (identical protocol for every model).

- loss: masked MAE in ORIGINAL flow units (missing targets ignored)
- Adam, gradient clipping, early stopping on validation MAE, best checkpoint restored
- resumable: a checkpoint is written atomically after every epoch, so a Colab disconnect
  only loses the current epoch; finished experiments are skipped unless --force
"""
import argparse
import os
import platform
import time

import numpy as np
import torch

from common import datetime_to_ns, CKPT_DIR, RES_DIR, SEED, load_json, save_json, set_seed
from data_utils import TrafficData, normalize_adj
from models import STGNN, count_params


def get_args(extra=None):
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--hidden", type=int, default=32, help="hidden dim d (16-32)")
    p.add_argument("--emb_dim", type=int, default=10, help="node-embedding dim d_e")
    p.add_argument("--topk", type=int, default=0, help="0 = dense adaptive graph, else keep top-k per row")
    p.add_argument("--alpha", type=float, default=None, help="fixed hybrid alpha; default = learned")
    p.add_argument("--train_stride", type=int, default=1, help=">1 subsamples training windows (CPU speed-up)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true", help="retrain even if results exist")
    if extra:
        extra(p)
    return p.parse_args()


def build_model(mode, data, args):
    A_auto, norm_mode = normalize_adj(data.adj, "auto")
    A_row, _ = normalize_adj(data.adj, "row")
    set_seed(args.seed)  # same initial encoder/head weights for every model
    model = STGNN(data.N, adj_mode=mode, A_fixed=A_auto, A_fixed_row=A_row, d=args.hidden,
                  emb_dim=args.emb_dim, topk=args.topk, alpha=args.alpha)
    return model.to(args.device), norm_mode


def masked_sums(pred, y, m):
    err = (pred - y) * m
    return err.abs().sum(dim=(0, 2)), (err ** 2).sum(dim=(0, 2)), m.sum(dim=(0, 2))  # per horizon


@torch.no_grad()
def evaluate(model, data, split, batch_size=128, keep_sensors=None):
    model.eval()
    starts = data.starts[split]
    abs_s = torch.zeros(4, device=data.device); sq_s = torch.zeros(4, device=data.device)
    cnt = torch.zeros(4, device=data.device)
    keep_p, keep_y, keep_m = [], [], []
    for i in range(0, len(starts), batch_size):
        x, y, m = data.batch(starts[i:i + batch_size])
        pred = model(x) * data.std + data.mean
        a, s, c = masked_sums(pred, y, m)
        abs_s += a; sq_s += s; cnt += c
        if keep_sensors is not None:
            keep_p.append(pred[:, :, keep_sensors].cpu()); keep_y.append(y[:, :, keep_sensors].cpu())
            keep_m.append(m[:, :, keep_sensors].cpu())
    mae_h = (abs_s / cnt).cpu().numpy(); rmse_h = torch.sqrt(sq_s / cnt).cpu().numpy()
    res = dict(MAE=float(abs_s.sum() / cnt.sum()), RMSE=float(torch.sqrt(sq_s.sum() / cnt.sum())),
               MAE_h=mae_h.tolist(), RMSE_h=rmse_h.tolist())
    if keep_sensors is not None:
        res["_preds"] = (torch.cat(keep_p).numpy(), torch.cat(keep_y).numpy(), torch.cat(keep_m).numpy())
    return res


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def run_experiment(name, mode, args, data=None, keep_sensors=None):
    """Train (or resume / reuse) one model and return its metrics dict."""
    res_name = f"metrics_{name}.json"
    done = load_json(res_name)
    if done is not None and not args.force:
        print(f"[skip] {name}: results already in {os.path.join(RES_DIR, res_name)} (use --force to retrain)")
        return done
    if data is None:
        data = TrafficData(args.device, args.train_stride)
    model, norm_mode = build_model(mode, data, args)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    last_ckpt = os.path.join(CKPT_DIR, f"{name}_last.pt")
    best_ckpt = os.path.join(CKPT_DIR, f"{name}_best.pt")
    state = dict(epoch=0, best_val=float("inf"), bad=0, history=[], train_time=0.0)
    if os.path.exists(last_ckpt) and not args.force:
        ck = torch.load(last_ckpt, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); state = ck["state"]
        print(f"[resume] {name} from epoch {state['epoch']}")

    print(f"\n### {name}: adj_mode={mode}, params={count_params(model):,}, device={args.device}, "
          f"train windows={len(data.starts['train'])}, fixed-adj normalisation={norm_mode}")
    tr = data.starts["train"]
    while state["epoch"] < args.epochs and state["bad"] < args.patience:
        ep = state["epoch"] + 1
        perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(args.seed + ep)).numpy()
        model.train()
        _sync(args.device); t0 = time.time()
        tot, n = 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            x, y, m = data.batch(tr[perm[i:i + args.batch_size]])
            pred = model(x) * data.std + data.mean
            loss = ((pred - y).abs() * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(x); n += len(x)
        _sync(args.device); ep_time = time.time() - t0
        state["train_time"] += ep_time
        val = evaluate(model, data, "val")
        rec = dict(epoch=ep, train_mae=tot / n, val_mae=val["MAE"], val_rmse=val["RMSE"], epoch_time=ep_time)
        if mode == "hybrid":
            rec["alpha"] = float(model.alpha())
        state["history"].append(rec)
        improved = val["MAE"] < state["best_val"] - 1e-4
        if improved:
            state["best_val"], state["bad"] = val["MAE"], 0
            torch.save(model.state_dict(), best_ckpt + ".tmp"); os.replace(best_ckpt + ".tmp", best_ckpt)
        else:
            state["bad"] += 1
        state["epoch"] = ep
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), state=state), last_ckpt + ".tmp")
        os.replace(last_ckpt + ".tmp", last_ckpt)
        print(f"  ep {ep:3d} | train MAE {rec['train_mae']:.3f} | val MAE {val['MAE']:.3f} RMSE {val['RMSE']:.3f}"
              f" | {ep_time:.1f}s" + (f" | alpha {rec['alpha']:.3f}" if mode == 'hybrid' else "") + (" *" if improved else ""))

    model.load_state_dict(torch.load(best_ckpt, map_location=args.device, weights_only=False))
    val = evaluate(model, data, "val")
    test = evaluate(model, data, "test", keep_sensors=keep_sensors)
    out = dict(name=name, adj_mode=mode, fixed_adj_normalisation=norm_mode if mode in ("fixed", "hybrid") else None,
               val=val, test={k: v for k, v in test.items() if k != "_preds"},
               train_time_s=state["train_time"], epochs_run=state["epoch"],
               best_epoch=int(np.argmin([h["val_mae"] for h in state["history"]]) + 1),
               mean_epoch_time_s=float(np.mean([h["epoch_time"] for h in state["history"]])),
               params=count_params(model), history=state["history"],
               hparams=dict(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, lr=args.lr,
                            weight_decay=args.weight_decay, hidden=args.hidden, emb_dim=args.emb_dim,
                            topk=args.topk, alpha=args.alpha, train_stride=args.train_stride, seed=args.seed,
                            in_len=8, out_len=4, loss="masked MAE", optimizer="Adam", grad_clip=5.0),
               device=str(args.device), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               cpu=platform.processor() or platform.machine())
    if mode == "hybrid":
        out["alpha"] = float(model.alpha())
    if mode in ("adaptive", "hybrid", "fixed"):
        with torch.no_grad():
            np.save(os.path.join(RES_DIR, f"adj_{name}.npy"), model.get_adj().cpu().numpy().astype(np.float32))
    if keep_sensors is not None:
        p, y, m = test["_preds"]
        np.savez_compressed(os.path.join(RES_DIR, f"preds_{name}.npz"), pred=p, true=y, mask=m,
                            sensors=np.array(keep_sensors),
                            target_time_h1=datetime_to_ns(data.target_times("test", 0)))
    save_json(out, res_name)
    print(f"  ==> {name}: val MAE {val['MAE']:.3f} RMSE {val['RMSE']:.3f} | test MAE {out['test']['MAE']:.3f} "
          f"RMSE {out['test']['RMSE']:.3f} | train time {state['train_time']/60:.1f} min")
    return out


def plot_sensors(data, n=6):
    """Fixed set of sensors whose test predictions are stored for the Q6 plots (best coverage)."""
    m = data.m[data.starts["test"][0]:].mean(0).cpu().numpy()
    order = np.argsort(-m)
    fwy = data.meta["Fwy"].astype(str).to_numpy() if "Fwy" in data.meta else np.arange(data.N).astype(str)
    chosen, seen = [], set()
    for i in order:                     # best-covered sensor on each of n different freeways
        if fwy[i] not in seen:
            chosen.append(int(i)); seen.add(fwy[i])
        if len(chosen) == n:
            break
    for i in order:                     # top up if fewer freeways than n
        if len(chosen) == n:
            break
        if int(i) not in chosen:
            chosen.append(int(i))
    return chosen
