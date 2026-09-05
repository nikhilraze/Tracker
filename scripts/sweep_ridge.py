"""Compare single-sample SGD against periodic refit on a rolling buffer.

Both are evaluated with the honest HORIZON-step label delay. Conclusion from
this script drives the design of neurotrack.models.OnlineForecaster.
"""

from __future__ import annotations

import itertools
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sweep import HORIZON, build_matrix

WARMUP = 300


def run_ridge(X, y, *, window, refit_every, alpha, min_fit=120):
    """Rolling-window Ridge, refit every `refit_every` samples.

    The buffer only ever contains pairs whose labels have already arrived, so
    there is no lookahead.
    """
    buf_X: deque = deque(maxlen=window)
    buf_y: deque = deque(maxlen=window)
    model = None
    scaler = None
    since_fit = 0
    err_m, err_n = [], []
    n = len(y)

    for i in range(n - HORIZON):
        anchor = y[i]
        truth = y[i + HORIZON]

        if model is not None:
            scaled = scaler.transform(X[i].reshape(1, -1))
            delta = float(model.predict(scaled)[0])
            pred = float(np.clip(anchor + delta, 0, 100))
            if i > WARMUP:
                err_m.append(abs(truth - pred))
                err_n.append(abs(truth - anchor))

        # label for step j has arrived now
        j = i - HORIZON
        if j >= 0:
            buf_X.append(X[j])
            buf_y.append(y[j + HORIZON] - y[j])
            since_fit += 1

        if len(buf_X) >= min_fit and (model is None or since_fit >= refit_every):
            data = np.asarray(buf_X)
            target = np.asarray(buf_y)
            scaler = StandardScaler().fit(data)
            model = Ridge(alpha=alpha).fit(scaler.transform(data), target)
            since_fit = 0

    if not err_m:
        return None, None, None
    mm = float(np.mean(err_m))
    nn = float(np.mean(err_n))
    return mm, nn, 1 - mm / nn


def main() -> int:
    X, cpu, ram = build_matrix()
    print(f"X {X.shape}   honest delay = {HORIZON} steps\n")

    rows = []
    for window, refit, alpha in itertools.product(
        [600, 1200, 2400], [30, 60, 120], [1.0, 10.0, 100.0, 1000.0]
    ):
        c = run_ridge(X, cpu, window=window, refit_every=refit, alpha=alpha)
        r = run_ridge(X, ram, window=window, refit_every=refit, alpha=alpha)
        if c[2] is None or r[2] is None:
            continue
        rows.append((min(c[2], r[2]), c[2], r[2], window, refit, alpha, c[0], c[1]))

    rows.sort(reverse=True)
    header = f"{'worst':>8} {'cpu':>8} {'ram':>8} {'window':>7} {'refit':>6} {'alpha':>7} {'mae':>7} {'naive':>7}"
    print(header)
    print("-" * len(header))
    for worst, c, r, window, refit, alpha, mae, naive in rows[:15]:
        print(f"{worst:+8.3f} {c:+8.3f} {r:+8.3f} {window:>7} {refit:>6} {alpha:>7} {mae:7.3f} {naive:7.3f}")

    best = rows[0]
    print(
        f"\nBEST ridge: window={best[3]} refit_every={best[4]} alpha={best[5]}"
        f"  -> cpu {best[1]:+.3f}, ram {best[2]:+.3f}"
    )
    print("\n(for reference, best honest single-sample SGD was cpu -0.007 / ram +0.070)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
