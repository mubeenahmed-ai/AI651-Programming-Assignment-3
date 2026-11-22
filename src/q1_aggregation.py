"""Q1 - What does a sum remember?

(a) Affine message phi(x) = w x + c  ->  sum = w*S_v + c*n_v
(b) Injective scalar message via base-(K+1) place values; brute-force check for K = 8
    plus the "count non-steady neighbours" message phi(x) = x^2
(c) Sum / Mean / Max with one-hot messages, verified by exhaustive enumeration of
    every neighbourhood with 1..8 neighbours.

Run:  python q1_aggregation.py
"""
import itertools

import numpy as np
import pandas as pd

from common import banner, save_json, save_text, savefig, plt, set_seed

set_seed()
TRENDS = (-1, 0, 1)


# --------------------------------------------------------------------------- (a)
def part_a():
    banner("Q1(a)  Affine message  phi(x) = w x + c")
    print("sum_{u in N(v)} (w x_u + c) = w * S_v + c * n_v   ->  h_v = rho(w S_v + c n_v)")
    print("The aggregate depends on the neighbourhood ONLY through (n_v, S_v), so two\n"
          "neighbourhoods with equal n_v and S_v give the same input to rho for every w, c, rho.")

    N1 = [0, 0]          # two steady neighbours
    N2 = [-1, +1]        # zero steady neighbours
    N3 = [-1, 0, 0, +1]  # bigger example: 2 steady
    N4 = [-1, -1, +1, +1]  # 0 steady
    pairs = [(N1, N2), (N3, N4)]

    rng = np.random.default_rng(0)
    rows = []
    for A, B in pairs:
        nA, SA, nB, SB = len(A), sum(A), len(B), sum(B)
        # random affine messages + random nonlinear rho (a tiny random MLP)
        max_diff = 0.0
        for _ in range(1000):
            w, c = rng.normal(size=2) * 3
            W1, b1, W2 = rng.normal(size=(8, 1)), rng.normal(size=8), rng.normal(size=8)
            rho = lambda z: W2 @ np.tanh(W1[:, 0] * z + b1)  # noqa: E731
            aggA = sum(w * x + c for x in A)
            aggB = sum(w * x + c for x in B)
            max_diff = max(max_diff, abs(rho(aggA) - rho(aggB)))
        rows.append(dict(N_first=str(A), N_second=str(B), n=f"{nA} / {nB}", S=f"{SA} / {SB}",
                         steady=f"{A.count(0)} / {B.count(0)}",
                         max_output_difference_over_1000_random_w_c_rho=max_diff))
        print(f"  {A} vs {B}: n={nA},{nB}  S={SA},{SB}  steady={A.count(0)},{B.count(0)}"
              f"  -> max |h_A - h_B| over 1000 random (w,c,rho) = {max_diff:.2e}")
    df = pd.DataFrame(rows)
    df.to_csv(__import__("os").path.join(__import__("common").RES_DIR, "q1a_pairs.csv"), index=False)
    return df


# --------------------------------------------------------------------------- (b)
def phi_b(K):
    B = K + 1
    return {-1: 1, 0: B, 1: B * B}


def decode(s, K):
    B = K + 1
    n_slow = s % B
    n_steady = (s // B) % B
    n_fast = s // (B * B)
    return n_slow, n_steady, n_fast


def part_b(K=8):
    banner(f"Q1(b)  Injective scalar message, K = {K}")
    B = K + 1
    phi = phi_b(K)
    print(f"Base B = K + 1 = {B}:  phi(-1) = 1,  phi(0) = B = {B},  phi(+1) = B^2 = {B*B}")
    print("Sum = n_slow*1 + n_steady*B + n_fast*B^2. Every count is <= K < B, so each count\n"
          "occupies its own base-B digit (no carries) -> decode with mod / integer division.")

    triples = [(a, b, c) for a in range(K + 1) for b in range(K + 1) for c in range(K + 1)
               if 1 <= a + b + c <= K]
    sums = np.array([a * phi[-1] + b * phi[0] + c * phi[1] for a, b, c in triples])
    unique = len(np.unique(sums)) == len(sums)
    decoded_ok = all(decode(int(s), K) == t for s, t in zip(sums, triples))
    print(f"  #valid count triples (1 <= n <= {K}): {len(triples)}")
    print(f"  all sums distinct: {unique}")
    print(f"  decode(sum) recovers (slow, steady, fast) for every triple: {decoded_ok}")

    # Also verify by summing actual neighbour messages (multisets of trends)
    ok_ms = True
    for n in range(1, K + 1):
        for ms in itertools.combinations_with_replacement(TRENDS, n):
            s = sum(phi[x] for x in ms)
            if decode(s, K) != (ms.count(-1), ms.count(0), ms.count(1)):
                ok_ms = False
    print(f"  verified on every neighbourhood multiset of size 1..{K}: {ok_ms}")

    # Show that a smaller base would fail (carries)
    Bbad = K
    phibad = {-1: 1, 0: Bbad, 1: Bbad ** 2}
    sbad = {}
    collision = None
    for a, b, c in triples:
        s = a + b * Bbad + c * Bbad ** 2
        if s in sbad and collision is None:
            collision = (sbad[s], (a, b, c), s)
        sbad[s] = (a, b, c)
    print(f"  sanity: base K = {K} is NOT enough, e.g. counts {collision[0]} and {collision[1]}"
          f" both sum to {collision[2]}")

    # Non-steady count
    print("\n  Non-steady count: phi(x) = x^2 (equivalently |x|): phi(-1)=1, phi(0)=0, phi(1)=1,")
    print("  so sum_u phi(x_u) = number of neighbours with trend -1 or +1, for ANY neighbourhood size.")
    ok_ns = all(sum(x * x for x in ms) == (ms.count(-1) + ms.count(1))
                for n in range(1, 30) for ms in itertools.combinations_with_replacement(TRENDS, n))
    print(f"  verified for neighbourhood sizes 1..29: {ok_ns}")

    # small figure: the code table
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.scatter(range(len(sums)), np.sort(sums), s=8)
    ax.set_xlabel("count triple (sorted by encoded sum)")
    ax.set_ylabel("encoded sum")
    ax.set_title(f"Q1(b): {len(triples)} count triples -> {len(np.unique(sums))} distinct sums (K={K}, base {B})")
    ax.grid(alpha=.3)
    savefig(fig, "q1b_encoding_injective")
    return dict(K=K, base=B, phi={str(k): v for k, v in phi.items()}, n_triples=len(triples),
                all_distinct=bool(unique), decode_ok=bool(decoded_ok), multiset_ok=bool(ok_ms),
                nonsteady_phi="x^2", nonsteady_ok=bool(ok_ns))


# --------------------------------------------------------------------------- (c)
ONEHOT = {-1: np.array([1, 0, 0]), 0: np.array([0, 1, 0]), 1: np.array([0, 0, 1])}


def agg(ms, how):
    M = np.stack([ONEHOT[x] for x in ms]).astype(float)
    if how == "Sum":
        return M.sum(0)
    if how == "Mean":
        return M.mean(0)
    return M.max(0)


TASKS = {
    "(i) Count steady neighbours": lambda ms: ms.count(0),
    "(ii) More than half slowing": lambda ms: ms.count(-1) > len(ms) / 2,
    "(iii) Any neighbour slowing": lambda ms: ms.count(-1) > 0,
}

# Proposed "Yes" decoders (rho) – these are verified exhaustively below
RHO = {
    ("(i) Count steady neighbours", "Sum"): (lambda a: int(round(a[1])), "rho = a2 (2nd coordinate = #steady)"),
    ("(ii) More than half slowing", "Sum"): (lambda a: a[0] > (a[0] + a[1] + a[2]) / 2,
                                            "n = a1+a2+a3 (sum of all coords); test a1 > n/2"),
    ("(ii) More than half slowing", "Mean"): (lambda a: a[0] > 0.5, "a1 = fraction slowing; test a1 > 1/2"),
    ("(iii) Any neighbour slowing", "Sum"): (lambda a: a[0] > 0, "test a1 > 0 (#slowing >= 1)"),
    ("(iii) Any neighbour slowing", "Mean"): (lambda a: a[0] > 0, "test a1 > 0 (fraction slowing > 0)"),
    ("(iii) Any neighbour slowing", "Max"): (lambda a: a[0] == 1, "test a1 = 1 (some neighbour flagged slowing)"),
}

# Hand-picked minimal counterexamples for the "No" cells (verified below)
COUNTER = {
    ("(i) Count steady neighbours", "Mean"): ([0], [0, 0]),
    ("(i) Count steady neighbours", "Max"): ([0], [0, 0]),
    ("(ii) More than half slowing", "Max"): ([-1, -1, 0], [-1, 0]),
}


def part_c(K=8):
    banner("Q1(c)  Sum vs Mean vs Max with one-hot messages (exhaustive check, 1..8 neighbours)")
    neighbourhoods = [list(ms) for n in range(1, K + 1)
                      for ms in itertools.combinations_with_replacement(TRENDS, n)]
    print(f"  enumerating {len(neighbourhoods)} distinct neighbourhoods (multisets)")
    rows = []
    for task, f in TASKS.items():
        for how in ("Sum", "Mean", "Max"):
            # Is the task a function of the aggregate?  group by aggregate value
            groups = {}
            for ms in neighbourhoods:
                key = tuple(np.round(agg(ms, how), 10))
                groups.setdefault(key, set()).add(f(ms))
            possible = all(len(v) == 1 for v in groups.values())
            if possible:
                rho, desc = RHO[(task, how)]
                ok = all(rho(agg(ms, how)) == f(ms) for ms in neighbourhoods)
                assert ok, (task, how)
                reason = desc + " [verified on all neighbourhoods]"
            else:
                A, B = COUNTER[(task, how)]
                va, vb = agg(A, how), agg(B, how)
                assert np.allclose(va, vb) and f(A) != f(B), (task, how)
                fmt = lambda v: "(" + ", ".join(f"{x:g}" for x in v) + ")"  # noqa: E731
                reason = (f"N1={A} and N2={B} both aggregate to {fmt(va)} but answers are "
                          f"{f(A)} vs {f(B)}")
            rows.append(dict(Task=task, Method=how, Possible="Yes" if possible else "No", Reason=reason))
            print(f"  {task:32s} {how:5s} -> {'Yes' if possible else 'No ':3s} | {reason}")
    df = pd.DataFrame(rows)
    import os
    from common import RES_DIR
    df.to_csv(os.path.join(RES_DIR, "q1c_table.csv"), index=False)
    return df, None


if __name__ == "__main__":
    da = part_a()
    rb = part_b(8)
    dc, md = part_c(8)
    try:
        md = dc.to_markdown(index=False)
    except Exception:
        md = dc.to_string(index=False)
    save_text(md, "q1c_table.md")
    save_json(dict(part_b=rb), "q1_summary.json")
    print("\nDone with Q1.")
