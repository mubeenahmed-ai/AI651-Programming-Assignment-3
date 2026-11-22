"""Q2 - Which way is the slowdown moving?  (40-sensor ring road)

(a)  symmetric layer  x_hat_i = a x_i + b (x_{i-1} + x_{i+1})
     i.  least-squares fit on one impulse (analytic + numeric)
     ii. 20-step autoregressive rollout of a Gaussian bump vs the true motion (heatmaps)
(b)  direction-aware layer  a x_i + w- x_{i-1} + w+ x_{i+1}  and the equivalent circular Conv1d
(c)  symmetric layer + history  a x_i + b (x_{i-1}+x_{i+1}) + c x_i(t-1)
     verified for backward AND forward motion.

Run:  python q2_direction.py
"""
import numpy as np

from common import banner, save_json, savefig, plt, set_seed

set_seed()
N = 40


def true_step(x, direction="backward"):
    """backward: x_i(t+1) = x_{i+1}(t)   (pattern moves against traffic, towards lower index)
       forward : x_i(t+1) = x_{i-1}(t)"""
    return np.roll(x, -1) if direction == "backward" else np.roll(x, 1)


def sym_layer(x, a, b):
    return a * x + b * (np.roll(x, 1) + np.roll(x, -1))  # roll(x,1)[i] = x_{i-1}, roll(x,-1)[i] = x_{i+1}


# --------------------------------------------------------------------------- (a)i
def part_a_i():
    banner("Q2(a)i  Fit a, b on one impulse at sensor 20")
    x = np.zeros(N); x[20] = 1.0
    y = true_step(x)  # 1 at sensor 19
    print("Predictions: x_hat_19 = b, x_hat_20 = a, x_hat_21 = b, others 0")
    print("Targets    : y_19 = 1,   y_20 = 0,   y_21 = 0")
    print("SSE(a,b) = (b-1)^2 + a^2 + b^2")
    print("dSSE/da = 2a = 0          -> a = 0")
    print("dSSE/db = 2(b-1) + 2b = 0 -> b = 1/2,   minimum SSE = 1/4 + 1/4 = 1/2")
    # numeric least squares: design matrix columns [x_i, x_{i-1}+x_{i+1}]
    X = np.stack([x, np.roll(x, 1) + np.roll(x, -1)], 1)
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    sse = np.sum((sym_layer(x, a, b) - y) ** 2)
    print(f"Numeric lstsq: a = {a:.6f}, b = {b:.6f}, SSE = {sse:.4f}")
    print("Sensors 19 and 21 receive the SAME prediction (0.5): the symmetric layer cannot tell\n"
          "which side the slowdown is moving to, so it hedges half-half.")
    return float(a), float(b), float(sse)


# --------------------------------------------------------------------------- (a)ii
def rollout(step_fn, x0, steps=20):
    xs = [x0]
    for _ in range(steps):
        xs.append(step_fn(xs[-1]))
    return np.stack(xs)


def heatmaps(seqs, titles, name, suptitle):
    vmin = min(s.min() for s in seqs); vmax = max(s.max() for s in seqs)
    fig, axes = plt.subplots(1, len(seqs), figsize=(5.2 * len(seqs), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s, t in zip(axes, seqs, titles):
        im = ax.imshow(s, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                       extent=[-0.5, N - 0.5, -0.5, s.shape[0] - 0.5])
        ax.set_xlabel("sensor index i"); ax.set_title(t)
    axes[0].set_ylabel("time step t")
    fig.colorbar(im, ax=axes.tolist(), label="slowdown $x_i(t)$")
    fig.suptitle(suptitle, y=1.03)
    return savefig(fig, name)


def part_a_ii(a, b):
    banner("Q2(a)ii  20-step rollout of a Gaussian bump")
    i = np.arange(N)
    x0 = np.exp(-(i - 20) ** 2 / 8)
    pred = rollout(lambda x: sym_layer(x, a, b), x0)
    true = rollout(true_step, x0)
    heatmaps([true, pred], ["True motion  $x_i(t+1)=x_{i+1}(t)$", f"Symmetric layer  a={a:.2f}, b={b:.2f}"],
             "q2a_heatmaps", "Q2(a)ii: bump propagation (same colour scale)")
    # quantitative description
    def centre(x):  # circular centre of mass
        ang = 2 * np.pi * i / N
        c = (np.angle(np.sum(x * np.exp(1j * ang))) % (2 * np.pi)) * N / (2 * np.pi)
        return round(float(c), 6) % N
    stats = []
    for t in (0, 5, 10, 20):
        stats.append(dict(t=t, true_centre=centre(true[t]), pred_centre=centre(pred[t]),
                          true_peak=true[t].max(), pred_peak=pred[t].max(),
                          true_mass=true[t].sum(), pred_mass=pred[t].sum()))
        print(f"  t={t:2d}: centre true={stats[-1]['true_centre']:.2f} pred={stats[-1]['pred_centre']:.2f} | "
              f"peak true={stats[-1]['true_peak']:.3f} pred={stats[-1]['pred_peak']:.3f} | "
              f"mass true={stats[-1]['true_mass']:.3f} pred={stats[-1]['pred_mass']:.3f}")
    # profile plot at t = 20
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(i, true[0], "k:", label="t=0")
    ax.plot(i, true[20], label="true, t=20")
    ax.plot(i, pred[20], label="symmetric layer, t=20")
    ax.set_xlabel("sensor index"); ax.set_ylabel("slowdown"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("Q2(a)ii: profiles at t = 20")
    savefig(fig, "q2a_profiles_t20")
    print("Observation: the true bump translates 20 sensors to the left (centre 20 -> 0) without\n"
          "changing shape; the symmetric layer keeps the bump centred at sensor 20 and only spreads\n"
          "it out (diffusion): the peak drops while the total mass is conserved (2b = 1).")
    return stats


# --------------------------------------------------------------------------- (b)
def part_b():
    banner("Q2(b)  Direction-aware weights")
    a, wm, wp = 0.0, 0.0, 1.0
    print("Choose a = 0, w- = 0, w+ = 1:  x_hat_i(t+1) = x_{i+1}(t) = x_i(t+1) exactly.")
    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(100):
        x = rng.normal(size=N)
        pred = a * x + wm * np.roll(x, 1) + wp * np.roll(x, -1)
        err = max(err, np.abs(pred - true_step(x)).max())
    print(f"  max |error| over 100 random snapshots: {err:.2e}")

    # 1D convolution implementation with circular padding
    kernel = np.array([wm, a, wp])  # multiplies [x_{i-1}, x_i, x_{i+1}]
    print(f"  Conv1d kernel for [x_(i-1), x_i, x_(i+1)] = {kernel.tolist()}  (cross-correlation, as in PyTorch)")
    print("  Ring boundary: circular padding of 1 on each side (x_-1 := x_39, x_40 := x_0).")
    conv_err = None
    try:
        import torch
        import torch.nn as nn
        conv = nn.Conv1d(1, 1, 3, padding=1, padding_mode="circular", bias=False)
        with torch.no_grad():
            conv.weight[:] = torch.tensor(kernel, dtype=torch.float32).view(1, 1, 3)
            x = torch.tensor(rng.normal(size=N), dtype=torch.float32)
            out = conv(x.view(1, 1, N)).view(-1).numpy()
        conv_err = float(np.abs(out - true_step(x.numpy())).max())
        print(f"  PyTorch Conv1d(padding_mode='circular') check, max |error| = {conv_err:.2e}")
    except ImportError:
        # numpy equivalent
        x = rng.normal(size=N)
        xp = np.concatenate([x[-1:], x, x[:1]])
        out = np.array([xp[i:i + 3] @ kernel for i in range(N)])
        conv_err = float(np.abs(out - true_step(x)).max())
        print(f"  NumPy circular-pad convolution check, max |error| = {conv_err:.2e}")
    i = np.arange(N)
    x0 = np.exp(-(i - 20) ** 2 / 8)
    pred = rollout(lambda x: a * x + wm * np.roll(x, 1) + wp * np.roll(x, -1), x0)
    true = rollout(true_step, x0)
    heatmaps([true, pred], ["True motion", "Direction-aware layer (0, 0, 1)"], "q2b_heatmaps",
             "Q2(b): direction-aware layer reproduces the motion exactly")
    return dict(a=a, w_minus=wm, w_plus=wp, max_err=float(err), conv_kernel=kernel.tolist(), conv_err=conv_err)


# --------------------------------------------------------------------------- (c)
def hist_layer(x_t, x_tm1, a, b, c):
    return a * x_t + b * (np.roll(x_t, 1) + np.roll(x_t, -1)) + c * x_tm1


def part_c():
    banner("Q2(c)  Symmetric neighbours + one step of history")
    a, b, c = 0.0, 1.0, -1.0
    print("Backward rule x_i(t) = x_{i+1}(t-1)  =>  x_i(t-1) = x_{i-1}(t)  (the old reading at i\n"
          "has moved one sensor back, to i-1).")
    print("Prediction = a x_i + b x_{i-1} + b x_{i+1} + c x_{i-1}   (substituting x_i(t-1) = x_{i-1}(t))")
    print("Target     = x_{i+1}(t)  ->  need a = 0, b = 1, b + c = 0  =>  (a, b, c) = (0, 1, -1)")
    print("Forward rule x_i(t+1) = x_{i-1}(t)  =>  x_i(t-1) = x_{i+1}(t).")
    print("Prediction = x_{i-1} + x_{i+1} - x_{i+1} = x_{i-1}(t) = target.  Same coefficients work!")

    rng = np.random.default_rng(1)
    res = {}
    for direction in ("backward", "forward"):
        err = 0.0
        for _ in range(100):
            x_prev = rng.normal(size=N)
            x_t = true_step(x_prev, direction)
            target = true_step(x_t, direction)
            err = max(err, np.abs(hist_layer(x_t, x_prev, a, b, c) - target).max())
        res[direction] = float(err)
        print(f"  {direction:8s} motion: max |error| over 100 random pattern pairs = {err:.2e}")

    # rollouts in both directions
    i = np.arange(N)
    x0 = np.exp(-(i - 20) ** 2 / 8)
    seqs, titles = [], []
    for direction in ("backward", "forward"):
        true = rollout(lambda x: true_step(x, direction), x0)
        xs = [x0, true_step(x0, direction)]  # needs one previous snapshot to start
        for _ in range(19):
            xs.append(hist_layer(xs[-1], xs[-2], a, b, c))
        seqs += [true, np.stack(xs)]
        titles += [f"True ({direction})", f"History layer ({direction})"]
    heatmaps(seqs, titles, "q2c_heatmaps", "Q2(c): one layer, equal neighbour weights, (a,b,c)=(0,1,-1)")
    print("Why it works: x_i(t-1) equals the CURRENT reading of the neighbour the pattern came from\n"
          "(upstream side). Subtracting it cancels that neighbour inside the symmetric sum, leaving only\n"
          "the neighbour the pattern is moving towards. The time difference encodes the direction.")
    return dict(a=a, b=b, c=c, max_err=res)


if __name__ == "__main__":
    a, b, sse = part_a_i()
    stats = part_a_ii(a, b)
    rb = part_b()
    rc = part_c()
    save_json(dict(part_a=dict(a=a, b=b, sse=sse, rollout_stats=stats), part_b=rb, part_c=rc), "q2_summary.json")
    print("\nDone with Q2.")
