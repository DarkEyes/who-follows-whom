"""Directed lagged coordination between per-person embedding sequences.

Per window, per person i: z_i in R^{T x D} (temporal token sequence).
  1. delta_i[t]   = z_i[t+1] - z_i[t]
  2. global-mode removal: delta_i <- delta_i - mean_j delta_j   (per t)
  3. directed lagged score:
        s(i -> j, tau) = mean_t cos(delta_i[t], delta_j[t + tau]),  tau >= 0
     "i leads j by tau" when tau > 0.
  4. calibration: circular-shift null on j's sequence -> z-score.

Reported per pair {i, j}: best directed score, its lag, direction, z-score.
"""

import numpy as np


def deltas(z: np.ndarray) -> np.ndarray:
    return z[1:] - z[:-1]


def remove_global_mode(delta_by_pid: dict) -> dict:
    stack = np.stack(list(delta_by_pid.values()))       # (P, T-1, D)
    mean = stack.mean(axis=0, keepdims=True)
    return {pid: d - mean[0] for pid, d in zip(delta_by_pid, stack)}


def _cos_seq(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine over aligned timesteps of two (t, D) sequences."""
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    return float((num / den).mean())


def lagged_score(di: np.ndarray, dj: np.ndarray, max_lag: int):
    """max over tau in [0, max_lag] of cos(di[t], dj[t+tau]).

    tau > 0 means i's change precedes j's (i leads).
    Returns (score, tau).
    """
    t = di.shape[0]
    best, best_tau = -np.inf, 0
    for tau in range(0, max_lag + 1):
        if t - tau < 3:
            break
        s = _cos_seq(di[: t - tau], dj[tau:])
        if s > best:
            best, best_tau = s, tau
    return best, best_tau


def directed_pair_score(di, dj, max_lag):
    """Evaluate both directions; return (score, lag, leader_first: bool)."""
    s_ij, tau_ij = lagged_score(di, dj, max_lag)   # i leads j
    s_ji, tau_ji = lagged_score(dj, di, max_lag)   # j leads i
    if s_ij >= s_ji:
        return s_ij, tau_ij, True
    return s_ji, tau_ji, False


def null_distribution(delta_by_pid: dict, max_lag: int, n_null: int,
                      rng: np.random.Generator):
    """Pooled circular-shift null across all pairs in this window."""
    pids = list(delta_by_pid)
    scores = []
    while len(scores) < n_null:
        i, j = rng.choice(len(pids), size=2, replace=False)
        dj = delta_by_pid[pids[j]]
        shift = rng.integers(1, dj.shape[0])
        dj_shifted = np.roll(dj, shift, axis=0)
        s, _ = lagged_score(delta_by_pid[pids[i]], dj_shifted, max_lag)
        scores.append(s)
    return np.asarray(scores)


def score_window(z_by_pid: dict, cfg, rng: np.random.Generator):
    """All pairwise directed scores for one window.

    Returns list of dicts sorted by z-score, descending.
    """
    d = {pid: deltas(z) for pid, z in z_by_pid.items()}
    if cfg.remove_global_mode and len(d) > 1:
        d = remove_global_mode(d)

    null = null_distribution(d, cfg.max_lag, cfg.n_null, rng)
    mu, sd = null.mean(), null.std() + 1e-8

    pids = sorted(d)
    directed = {}
    for i in pids:
        for j in pids:
            if i == j:
                continue
            s, tau = lagged_score(d[i], d[j], cfg.max_lag)
            directed[f"{i}->{j}"] = {
                "score": round(float(s), 4),
                "lag_steps": int(tau),
                "z": round(float((s - mu) / sd), 3),
            }

    rows = []
    for a in range(len(pids)):
        for b in range(a + 1, len(pids)):
            i, j = pids[a], pids[b]
            e_ij, e_ji = directed[f"{i}->{j}"], directed[f"{j}->{i}"]
            e, leader, follower = ((e_ij, i, j) if e_ij["z"] >= e_ji["z"]
                                   else (e_ji, j, i))
            rows.append({
                "pair": [i, j],
                "leader": leader,
                "follower": follower,
                "lag_steps": e["lag_steps"],
                "score": e["score"],
                "z": e["z"],
            })
    rows.sort(key=lambda r: -r["z"])
    return rows, directed


def aggregate_network(window_results: list) -> dict:
    """Directed following weights F[leader -> follower] = sum of positive z."""
    net = {}
    for w in window_results:
        for e in w["edges"]:
            if e["z"] <= 0:
                continue
            key = f'{e["leader"]}->{e["follower"]}'
            net[key] = round(net.get(key, 0.0) + e["z"], 3)
    return dict(sorted(net.items(), key=lambda kv: -kv[1]))
