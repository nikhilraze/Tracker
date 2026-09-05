"""How far ahead is CPU/RAM actually predictable?

The feature vector only looks backwards, so the same matrix can be scored
against several horizons. This decides the default forecast_horizon and shows
where persistence stops being unbeatable.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sweep import build_matrix

WARMUP = 300


def run_ridge(X, y, horizon, *, window=2400, refit_every=30, alpha=100.0, min_fit=120):
    buf_X: deque = deque(maxlen=window)
    buf_y: deque = deque(maxlen=window)
    model = scaler = None
    since_fit = 0
    err_m, err_n = [], []

    for i in range(len(y) - horizon):
        anchor, truth = y[i], y[i + horizon]
        if model is not None:
            delta = float(model.predict(scaler.transform(X[i].reshape(1, -1)))[0])
            pred = float(np.clip(anchor + delta, 0, 100))
            if i > WARMUP:
                err_m.append(abs(truth - pred))
                err_n.append(abs(truth - anchor))
        j = i - horizon
        if j >= 0:
            buf_X.append(X[j])
            buf_y.append(y[j + horizon] - y[j])
            since_fit += 1
        if len(buf_X) >= min_fit and (model is None or since_fit >= refit_every):
            data, target = np.asarray(buf_X), np.asarray(buf_y)
            scaler = StandardScaler().fit(data)
            model = Ridge(alpha=alpha).fit(scaler.transform(data), target)
            since_fit = 0

    if not err_m:
        return None, None, None
    mm, nn = float(np.mean(err_m)), float(np.mean(err_n))
    return mm, nn, 1 - mm / nn


def main() -> int:
    X, cpu, ram = build_matrix()
    print("Rolling Ridge (window=2400, refit=30, alpha=100), honest label delay\n")
    header = f"{'H':>4} {'cpu skill':>10} {'cpu mae':>9} {'cpu naive':>10}   {'ram skill':>10} {'ram mae':>9} {'ram naive':>10}"
    print(header)
    print("-" * len(header))
    for horizon in (3, 5, 10, 15, 30, 60, 120):
        cm, cn, cs = run_ridge(X, cpu, horizon)
        rm, rn, rs = run_ridge(X, ram, horizon)
        print(
            f"{horizon:>4} {cs:+10.3f} {cm:9.3f} {cn:10.3f}   {rs:+10.3f} {rm:9.3f} {rn:10.3f}"
        )
    print("\nskill > 0 means the learned model beats 'it will stay the same'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
