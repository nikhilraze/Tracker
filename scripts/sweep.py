"""Diagnostic sweep to choose forecaster hyperparameters empirically.

Builds the feature matrix once, caches it, then replays the same online training
loop under different configurations and reports skill vs the naive baseline.
Not part of the app; kept because it documents *why* the defaults are what
they are.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from neurotrack.features import build_forecast_features

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_models import make_trace

CACHE = Path("/tmp/nt_feat_cache.npz")
HORIZON = 15
N = 4000


def build_matrix():
    if CACHE.exists():
        data = np.load(CACHE)
        return data["X"], data["cpu"], data["ram"]
    trace = make_trace(N)
    rows, cpu, ram = [], [], []
    history = []
    for sample in trace:
        history.append(sample)
        if len(history) > 3600:
            history.pop(0)
        rows.append(build_forecast_features(history, len(history) - 1))
        cpu.append(sample.cpu)
        ram.append(sample.ram)
    X = np.asarray(rows)
    cpu = np.asarray(cpu)
    ram = np.asarray(ram)
    np.savez(CACHE, X=X, cpu=cpu, ram=ram)
    return X, cpu, ram


class RunningY:
    """Running mean/std of the target, so eta0 is scale-free across metrics."""

    def __init__(self, floor=0.5):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.floor = floor

    def update(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    @property
    def std(self):
        if self.n < 2:
            return self.floor
        return max(self.floor, math.sqrt(self.m2 / (self.n - 1)))


def run(X, y, *, mode, loss, eta0, average, alpha, epsilon=5.0, clip=5.0,
        scale_y=False, delay=HORIZON, lags=None):
    """Replay the online loop; returns (model_mae, naive_mae, skill).

    ``delay`` is critical for honesty. A label for a prediction made at step i is
    only knowable at step i+HORIZON, so training must lag by that much. Using
    delay=1 (train on step i's label immediately) leaks future values and
    inflated measured skill from -0.35 to +0.41 in early runs of this script.
    """
    scaler = StandardScaler()
    model = SGDRegressor(
        loss=loss,
        epsilon=epsilon,
        penalty="l2",
        alpha=alpha,
        learning_rate="constant",
        eta0=eta0,
        average=average,
        random_state=0,
    )
    n = len(y)
    started = False
    err_m, err_n = [], []
    ystat = RunningY()

    cols = slice(None) if lags is None else lags

    for i in range(n - HORIZON):
        feats = X[i].reshape(1, -1)[:, cols]
        anchor = y[i]
        truth = y[i + HORIZON]

        # predict first (as live: model has not seen this label yet)
        if started:
            scaled = np.clip(
                np.nan_to_num(scaler.transform(feats), nan=0.0, posinf=0.0, neginf=0.0),
                -clip,
                clip,
            )
            raw = float(model.predict(scaled)[0])
            if scale_y:
                raw = raw * ystat.std + ystat.mean
            pred = raw if mode == "abs" else anchor + raw
            pred = float(np.clip(pred, 0, 100))
            # only score after a warmup so startup transients do not dominate
            if i > 300:
                err_m.append(abs(truth - pred))
                err_n.append(abs(truth - anchor))

        # Train on the oldest pair whose label has legitimately arrived.
        j = i - delay + 1
        if j >= 0:
            jfeats = X[j].reshape(1, -1)[:, cols]
            janchor = y[j]
            jtruth = y[j + HORIZON]
            scaler.partial_fit(jfeats)
            started = True
            scaled = np.clip(
                np.nan_to_num(scaler.transform(jfeats), nan=0.0, posinf=0.0, neginf=0.0),
                -clip,
                clip,
            )
            target = jtruth if mode == "abs" else jtruth - janchor
            ystat.update(float(target))
            fit_target = (float(target) - ystat.mean) / ystat.std if scale_y else float(target)
            model.partial_fit(scaled, np.asarray([fit_target]))

    if not err_m:
        return None, None, None
    mae_m = float(np.mean(err_m))
    mae_n = float(np.mean(err_n))
    return mae_m, mae_n, 1 - mae_m / mae_n


def main() -> int:
    X, cpu, ram = build_matrix()
    print(f"feature matrix {X.shape}\n")

    # Score on BOTH targets and rank by the worse of the two, because one set of
    # defaults has to serve a spiky metric (CPU) and a smooth one (RAM).
    results = []
    for mode, loss, eta0, scale_y in itertools.product(
        ["delta", "abs"],
        ["squared_error", "huber"],
        [0.001, 0.003, 0.01, 0.03, 0.1],
        [True, False],
    ):
        cpu_m, cpu_n, cpu_skill = run(
            X, cpu, mode=mode, loss=loss, eta0=eta0, average=False, alpha=1e-4, scale_y=scale_y
        )
        ram_m, ram_n, ram_skill = run(
            X, ram, mode=mode, loss=loss, eta0=eta0, average=False, alpha=1e-4, scale_y=scale_y
        )
        if cpu_skill is None or ram_skill is None:
            continue
        results.append((min(cpu_skill, ram_skill), cpu_skill, ram_skill, mode, loss, eta0, scale_y))

    results.sort(reverse=True, key=lambda r: r[0])
    header = f"{'worst':>8} {'cpu':>8} {'ram':>8} {'mode':>6} {'loss':>14} {'eta0':>7} {'scaleY':>7}"
    print(header)
    print("-" * len(header))
    for worst, cs, rs, mode, loss, eta0, scale_y in results[:20]:
        print(f"{worst:+8.3f} {cs:+8.3f} {rs:+8.3f} {mode:>6} {loss:>14} {eta0:>7} {scale_y!s:>7}")

    best = results[0]
    print(
        f"\nBEST: mode={best[3]} loss={best[4]} eta0={best[5]} scale_y={best[6]}"
        f"  -> cpu skill {best[1]:+.3f}, ram skill {best[2]:+.3f}"
    )
    return 0





def focused() -> int:
    """Second stage: lock in eta0/epsilon for delta + scaled target."""
    X, cpu, ram = build_matrix()
    print("delta mode, scale_y=True, average=False, alpha=1e-4\n")
    header = f"{'worst':>8} {'cpu':>8} {'ram':>8} {'eta0':>7} {'eps':>6}"
    print(header)
    print("-" * len(header))
    rows = []
    for eta0 in [0.005, 0.01, 0.02, 0.03, 0.05]:
        for eps in [0.5, 1.0, 2.0, 5.0, 10.0]:
            c = run(X, cpu, mode="delta", loss="huber", eta0=eta0, average=False,
                    alpha=1e-4, epsilon=eps, scale_y=True)
            r = run(X, ram, mode="delta", loss="huber", eta0=eta0, average=False,
                    alpha=1e-4, epsilon=eps, scale_y=True)
            rows.append((min(c[2], r[2]), c[2], r[2], eta0, eps))
    rows.sort(reverse=True)
    for worst, c, r, eta0, eps in rows:
        print(f"{worst:+8.3f} {c:+8.3f} {r:+8.3f} {eta0:>7} {eps:>6}")
    best = rows[0]
    print(f"\nBEST: eta0={best[3]} epsilon={best[4]} -> cpu {best[1]:+.3f} ram {best[2]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(focused() if "--focused" in sys.argv else main())
