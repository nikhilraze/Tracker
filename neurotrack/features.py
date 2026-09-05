"""Feature engineering.

The forecaster and the anomaly detector are both fed from here, so training and
live inference are guaranteed to build identical vectors. If you add a feature,
add it to the name tuple and the builder in the same edit - the models key their
persisted scalers off ``len(FEATURE_NAMES)`` and will refuse to load a stale
model whose width no longer matches.

Two distinct vectors are produced:

``build_forecast_features``
    Rich, lag-heavy representation used to predict a metric H seconds ahead.
    Autoregressive lags plus rolling statistics let a linear model capture both
    the current level and its momentum.

``build_anomaly_features``
    Small, stable representation describing "what the machine is doing right
    now" across several resources at once. Kept low-dimensional because
    IsolationForest is trained on a rolling window and must stay fast.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .collector import Sample

# Lags (in samples) fed to the forecaster. At 1 Hz these are seconds.
LAGS: tuple[int, ...] = (1, 2, 3, 5, 8, 15, 30, 60)
# Rolling window lengths (in samples) for mean/std features.
WINDOWS: tuple[int, ...] = (5, 15, 60)

FEATURE_NAMES: tuple[str, ...] = (
    # current state
    # NOTE: deliberately excludes disk_pct and uptime. Both drift monotonically,
    # and standardising a monotonic feature turns it into a proxy for "time
    # index", which a linear model happily overfits to.
    "cpu",
    "cpu_max_core",
    "ram",
    "swap",
    "disk_io_mbps",
    "net_io_mbps",
    "load1_per_core",
    "proc_count_scaled",
    # autoregressive history of the two volatile metrics
    *[f"cpu_lag{k}" for k in LAGS],
    *[f"ram_lag{k}" for k in LAGS],
    # rolling statistics
    *[f"cpu_mean{w}" for w in WINDOWS],
    *[f"cpu_std{w}" for w in WINDOWS],
    *[f"ram_mean{w}" for w in WINDOWS],
    *[f"ram_std{w}" for w in WINDOWS],
    # momentum / deviation from the recent local baseline
    "cpu_delta1",
    "cpu_delta5",
    "cpu_dev_mean15",
    "ram_delta1",
    "ram_delta5",
    "ram_dev_mean15",
    "io_delta1",
    # temporal context: daily and weekly rhythm
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # constant offset term
    "bias",
)

ANOMALY_FEATURE_NAMES: tuple[str, ...] = (
    "cpu",
    "cpu_max_core",
    "ram",
    "swap",
    "disk_io_mbps",
    "net_io_mbps",
    "load1_per_core",
    "cpu_std15",
    "cpu_dev_mean60",
    "ram_dev_mean60",
)

N_FEATURES = len(FEATURE_NAMES)
N_ANOMALY_FEATURES = len(ANOMALY_FEATURE_NAMES)

# History needed before a full-width feature vector can be built.
MIN_HISTORY = max(max(LAGS), max(WINDOWS)) + 1


def _safe(value: float | None, default: float = 0.0) -> float:
    if value is None:
        return default
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _at(history: Sequence[Sample], idx: int, lag: int) -> Sample:
    """Sample ``lag`` steps before ``idx``, clamped to the start of history."""
    return history[max(0, idx - lag)]


def _window(history: Sequence[Sample], idx: int, size: int) -> list[Sample]:
    start = max(0, idx - size + 1)
    return list(history[start : idx + 1])


def build_forecast_features(history: Sequence[Sample], idx: int | None = None) -> np.ndarray:
    """Build the forecaster's feature vector for position ``idx``.

    Short history is tolerated by clamping lags to the oldest sample, so the
    model can start learning immediately instead of waiting a full minute.
    """
    if not history:
        raise ValueError("history is empty")
    idx = len(history) - 1 if idx is None else idx
    if idx < 0 or idx >= len(history):
        raise IndexError(f"idx {idx} out of range for history of {len(history)}")

    cur = history[idx]
    values: list[float] = [
        _safe(cur.cpu),
        _safe(cur.cpu_max_core),
        _safe(cur.ram),
        _safe(cur.swap),
        _safe(cur.disk_io_mbps),
        _safe(cur.net_io_mbps),
        _safe(cur.load1_per_core),
        _safe(cur.proc_count) / 100.0,
    ]

    for lag in LAGS:
        values.append(_safe(_at(history, idx, lag).cpu))
    for lag in LAGS:
        values.append(_safe(_at(history, idx, lag).ram))

    cpu_windows: dict[int, np.ndarray] = {}
    ram_windows: dict[int, np.ndarray] = {}
    for size in WINDOWS:
        chunk = _window(history, idx, size)
        cpu_windows[size] = np.array([_safe(s.cpu) for s in chunk], dtype=float)
        ram_windows[size] = np.array([_safe(s.ram) for s in chunk], dtype=float)

    for size in WINDOWS:
        values.append(float(cpu_windows[size].mean()))
    for size in WINDOWS:
        values.append(float(cpu_windows[size].std()) if cpu_windows[size].size > 1 else 0.0)
    for size in WINDOWS:
        values.append(float(ram_windows[size].mean()))
    for size in WINDOWS:
        values.append(float(ram_windows[size].std()) if ram_windows[size].size > 1 else 0.0)

    cpu_now = _safe(cur.cpu)
    ram_now = _safe(cur.ram)
    values.append(cpu_now - _safe(_at(history, idx, 1).cpu))
    values.append(cpu_now - _safe(_at(history, idx, 5).cpu))
    values.append(cpu_now - float(cpu_windows[15].mean()))
    values.append(ram_now - _safe(_at(history, idx, 1).ram))
    values.append(ram_now - _safe(_at(history, idx, 5).ram))
    values.append(ram_now - float(ram_windows[15].mean()))
    values.append(_safe(cur.disk_io_mbps) - _safe(_at(history, idx, 1).disk_io_mbps))

    # Cyclical time encoding: sin/cos keeps 23:59 adjacent to 00:00.
    local = _local_time_parts(cur.ts)
    hour_frac, dow = local
    values.append(math.sin(2 * math.pi * hour_frac / 24.0))
    values.append(math.cos(2 * math.pi * hour_frac / 24.0))
    values.append(math.sin(2 * math.pi * dow / 7.0))
    values.append(math.cos(2 * math.pi * dow / 7.0))

    values.append(1.0)

    vector = np.asarray(values, dtype=float)
    if vector.shape[0] != N_FEATURES:
        raise AssertionError(
            f"feature width mismatch: built {vector.shape[0]}, expected {N_FEATURES}"
        )
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def build_anomaly_features(history: Sequence[Sample], idx: int | None = None) -> np.ndarray:
    """Compact multivariate 'current behaviour' vector for anomaly scoring."""
    if not history:
        raise ValueError("history is empty")
    idx = len(history) - 1 if idx is None else idx
    cur = history[idx]

    cpu15 = np.array([_safe(s.cpu) for s in _window(history, idx, 15)], dtype=float)
    cpu60 = np.array([_safe(s.cpu) for s in _window(history, idx, 60)], dtype=float)
    ram60 = np.array([_safe(s.ram) for s in _window(history, idx, 60)], dtype=float)

    values = [
        _safe(cur.cpu),
        _safe(cur.cpu_max_core),
        _safe(cur.ram),
        _safe(cur.swap),
        _safe(cur.disk_io_mbps),
        _safe(cur.net_io_mbps),
        _safe(cur.load1_per_core),
        float(cpu15.std()) if cpu15.size > 1 else 0.0,
        _safe(cur.cpu) - float(cpu60.mean()),
        _safe(cur.ram) - float(ram60.mean()),
    ]
    vector = np.asarray(values, dtype=float)
    if vector.shape[0] != N_ANOMALY_FEATURES:
        raise AssertionError(
            f"anomaly feature width mismatch: built {vector.shape[0]},"
            f" expected {N_ANOMALY_FEATURES}"
        )
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def _local_time_parts(ts: float) -> tuple[float, int]:
    """(hour-of-day as a float, weekday index) in the machine's local zone."""
    import time as _time

    parts = _time.localtime(ts)
    hour_frac = parts.tm_hour + parts.tm_min / 60.0 + parts.tm_sec / 3600.0
    return hour_frac, parts.tm_wday


@dataclass
class RobustZTracker:
    """Streaming robust z-score for one metric.

    Uses exponentially weighted mean and mean-absolute-deviation instead of a
    standard deviation, so a single huge spike does not inflate the threshold
    and mask everything that follows. Provides an instant anomaly signal that
    works from ~30 samples, long before IsolationForest has enough data.
    """

    alpha: float = 0.01
    mean: float = 0.0
    mad: float = 0.0
    count: int = 0

    def update(self, value: float) -> float:
        value = _safe(value)
        self.count += 1
        if self.count == 1:
            self.mean = value
            self.mad = 0.0
            return 0.0
        deviation = value - self.mean
        # Warm up faster than `alpha` during the first samples.
        alpha = max(self.alpha, 1.0 / self.count)
        self.mean += alpha * deviation
        self.mad += alpha * (abs(deviation) - self.mad)
        return self.score(value)

    def score(self, value: float) -> float:
        """Signed robust z-score. 1.4826 scales MAD to a std-equivalent."""
        if self.count < 10 or self.mad <= 1e-9:
            return 0.0
        return (_safe(value) - self.mean) / (1.4826 * self.mad)

    def state(self) -> dict:
        return {"mean": self.mean, "mad": self.mad, "count": self.count, "alpha": self.alpha}

    @classmethod
    def from_state(cls, state: dict) -> RobustZTracker:
        tracker = cls(alpha=state.get("alpha", 0.01))
        tracker.mean = state.get("mean", 0.0)
        tracker.mad = state.get("mad", 0.0)
        tracker.count = state.get("count", 0)
        return tracker
