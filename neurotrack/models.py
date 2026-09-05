"""The learning layer.

Two models run side by side. Neither ships with pretrained weights: both are
trained entirely on the machine they run on, from the samples this app collects.

1. ``ForecastService`` - supervised regression, trained *online*.
   Predicts CPU and RAM ``horizon`` seconds into the future. Labels are obtained
   for free by waiting: a prediction made at time ``t`` for ``t+H`` is scored
   against reality once the sample at ``t+H`` arrives, and that pair becomes one
   training example via ``SGDRegressor.partial_fit``. No human labelling, no
   batch retraining, and the model keeps adapting as usage patterns change.

2. ``AnomalyModel`` - unsupervised, two layers.
   * a streaming robust z-score per metric, useful within ~30 seconds;
   * an ``IsolationForest`` refit on a rolling window, which catches abnormal
     *combinations* of metrics that per-metric thresholds miss.

Every model reports its own accuracy against a naive baseline so the UI can
state honestly whether the prediction is worth trusting yet.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import (
    ANOMALY_FEATURE_NAMES,
    FEATURE_NAMES,
    N_ANOMALY_FEATURES,
    N_FEATURES,
    RobustZTracker,
)

MODEL_FORMAT_VERSION = 3


class RunningMoments:
    """Welford running mean/variance with a floor on the standard deviation.

    Used to standardise the *label*. Without it a single learning rate cannot
    serve both CPU (spiky, large changes) and RAM (smooth, tiny changes): the
    rate that converges for one diverges for the other. Standardising the target
    makes the learning rate scale-free, so the same defaults work for any metric
    on any machine.
    """

    def __init__(self, floor: float = 0.5) -> None:
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0
        self.floor = floor

    def update(self, value: float) -> None:
        value = float(value)
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return self.floor
        variance = self._m2 / (self.n - 1)
        if variance <= 0:
            return self.floor
        return max(self.floor, float(np.sqrt(variance)))

    def standardize(self, value: float) -> float:
        return (float(value) - self.mean) / self.std

    def restore(self, value: float) -> float:
        return float(value) * self.std + self.mean

    def state(self) -> dict:
        return {"n": self.n, "mean": self.mean, "m2": self._m2, "floor": self.floor}

    @classmethod
    def from_state(cls, state: dict) -> RunningMoments:
        obj = cls(floor=state.get("floor", 0.5))
        obj.n = state.get("n", 0)
        obj.mean = state.get("mean", 0.0)
        obj._m2 = state.get("m2", 0.0)
        return obj


# ---------------------------------------------------------------------------
# online error tracking
# ---------------------------------------------------------------------------
@dataclass
class OnlineError:
    """Exponentially weighted error, compared against a naive baseline.

    The baseline is *persistence*: "the value in H seconds equals the value
    now". Beating it is the only meaningful definition of the model working, and
    for smooth signals it is a surprisingly strong opponent.
    """

    decay: float = 0.995
    model_weighted: float = 0.0
    naive_weighted: float = 0.0
    weight: float = 0.0
    count: int = 0
    last_error: float | None = None

    def update(self, model_abs_err: float, naive_abs_err: float) -> None:
        self.count += 1
        self.weight = self.weight * self.decay + 1.0
        self.model_weighted = self.model_weighted * self.decay + model_abs_err
        self.naive_weighted = self.naive_weighted * self.decay + naive_abs_err
        self.last_error = model_abs_err

    @property
    def model_mae(self) -> float | None:
        return self.model_weighted / self.weight if self.weight > 0 else None

    @property
    def naive_mae(self) -> float | None:
        return self.naive_weighted / self.weight if self.weight > 0 else None

    @property
    def skill(self) -> float | None:
        """1 - MAE_model / MAE_naive. Positive means it beats persistence."""
        model, naive = self.model_mae, self.naive_mae
        if model is None or naive is None or naive <= 1e-9:
            return None
        return 1.0 - (model / naive)

    def state(self) -> dict:
        return {
            "decay": self.decay,
            "model_weighted": self.model_weighted,
            "naive_weighted": self.naive_weighted,
            "weight": self.weight,
            "count": self.count,
        }

    @classmethod
    def from_state(cls, state: dict) -> OnlineError:
        obj = cls(decay=state.get("decay", 0.995))
        obj.model_weighted = state.get("model_weighted", 0.0)
        obj.naive_weighted = state.get("naive_weighted", 0.0)
        obj.weight = state.get("weight", 0.0)
        obj.count = state.get("count", 0)
        return obj


# ---------------------------------------------------------------------------
# forecasting
# ---------------------------------------------------------------------------
class OnlineForecaster:
    """Incrementally retrained regressor for one target metric.

    Design notes, all of them measured rather than assumed (see
    ``scripts/sweep*.py``):

    **It learns the change, not the level.** The label is ``y[t+H] - y[t]`` and
    the forecast is ``y[t] + predicted_change``. The naive baseline ("it will be
    the same in H seconds") is exactly ``change = 0``, so a model with zero
    weights *equals* the baseline. Learning can only add value instead of having
    to rediscover it, and the target stays centred and roughly stationary.

    **Rolling-window Ridge, not single-sample SGD.** Per-sample ``partial_fit``
    was tried first and lost to the baseline: consecutive samples are almost
    identical, so each gradient step is dominated by noise, and every learning
    rate either crawled or oscillated. Refitting a closed-form Ridge on a buffer
    of recent labelled pairs has no learning rate, is far more stable, and costs
    only a few milliseconds every ``refit_every`` samples.

    **Adaptive shrinkage toward persistence.** Some signals genuinely are not
    forecastable - second-scale CPU is close to a random walk, while RAM trends
    predictably. The emitted forecast is
    ``anchor + weight * predicted_change`` where ``weight`` is driven by the
    model's own recently measured skill. If the model is not beating persistence
    the weight decays to 0 and the output degrades to the baseline, so a
    hard-to-predict metric cannot produce confidently wrong advice.
    """

    # Skill at which the model is trusted completely.
    FULL_TRUST_SKILL = 0.10
    # Graded predictions required before skill is considered meaningful.
    MIN_GRADED_FOR_TRUST = 60

    def __init__(
        self,
        name: str,
        horizon: int,
        window: int = 2400,
        refit_every: int = 30,
        alpha: float = 100.0,
        min_fit: int = 120,
        decay: float = 0.995,
    ) -> None:
        from sklearn.preprocessing import StandardScaler

        self.name = name
        self.horizon = horizon
        self.window = window
        self.refit_every = refit_every
        self.alpha = alpha
        self.min_fit = min_fit

        self._buf_X: deque[np.ndarray] = deque(maxlen=window)
        self._buf_y: deque[float] = deque(maxlen=window)
        self._scaler_cls = StandardScaler
        self.scaler = None
        self.model = None
        self._since_fit = 0
        self.fit_count = 0
        self.last_fit_ts: float | None = None

        # Skill of the raw model output - drives shrinkage and honest reporting.
        self.error = OnlineError(decay=decay)
        # Skill of what is actually emitted after shrinkage.
        self.effective_error = OnlineError(decay=decay)
        self.samples_trained = 0

    # -- training --------------------------------------------------------
    def _refit(self) -> None:
        from sklearn.linear_model import Ridge

        data = np.asarray(self._buf_X, dtype=float)
        target = np.asarray(self._buf_y, dtype=float)
        if data.shape[0] < self.min_fit:
            return
        scaler = self._scaler_cls().fit(data)
        model = Ridge(alpha=self.alpha).fit(scaler.transform(data), target)
        self.scaler = scaler
        self.model = model
        self._since_fit = 0
        self.fit_count += 1
        self.last_fit_ts = time.time()

    def learn(self, features: np.ndarray, anchor: float, future_value: float) -> None:
        """Record one delayed label and refit when due.

        ``anchor`` is the metric value when the prediction was made;
        ``future_value`` is what it actually became H seconds later.
        """
        self._buf_X.append(np.asarray(features, dtype=float))
        self._buf_y.append(float(future_value) - float(anchor))
        self.samples_trained += 1
        self._since_fit += 1

        if len(self._buf_X) >= self.min_fit and (
            self.model is None or self._since_fit >= self.refit_every
        ):
            self._refit()

    # -- inference -------------------------------------------------------
    @property
    def trust_weight(self) -> float:
        """0..1 blend weight between persistence (0) and the model (1)."""
        if self.error.count < self.MIN_GRADED_FOR_TRUST:
            return 0.0
        skill = self.error.skill
        if skill is None or skill <= 0:
            return 0.0
        return float(np.clip(skill / self.FULL_TRUST_SKILL, 0.0, 1.0))

    def predict_parts(
        self, features: np.ndarray, anchor: float
    ) -> tuple[float | None, float | None, float]:
        """Return (emitted forecast, raw model forecast, trust weight)."""
        if self.model is None or self.scaler is None:
            return None, None, 0.0
        try:
            scaled = self.scaler.transform(np.asarray(features, dtype=float).reshape(1, -1))
            delta = float(self.model.predict(scaled)[0])
        except Exception:
            return None, None, 0.0
        if not np.isfinite(delta):
            return None, None, 0.0

        anchor = float(anchor)
        raw = float(np.clip(anchor + delta, 0.0, 100.0))
        weight = self.trust_weight
        blended = float(np.clip(anchor + weight * delta, 0.0, 100.0))
        return blended, raw, weight

    def predict(self, features: np.ndarray, anchor: float) -> float | None:
        return self.predict_parts(features, anchor)[0]

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def top_weights(self, k: int = 12) -> list[tuple[str, float]]:
        """Largest-magnitude coefficients, for the model inspection page."""
        if self.model is None or not hasattr(self.model, "coef_"):
            return []
        coefs = np.asarray(self.model.coef_, dtype=float).ravel()
        if coefs.shape[0] != len(FEATURE_NAMES):
            return []
        order = np.argsort(np.abs(coefs))[::-1][:k]
        return [(FEATURE_NAMES[i], float(coefs[i])) for i in order]

    # -- persistence -----------------------------------------------------
    def state(self) -> dict:
        return {
            "name": self.name,
            "horizon": self.horizon,
            "window": self.window,
            "refit_every": self.refit_every,
            "alpha": self.alpha,
            "min_fit": self.min_fit,
            "scaler": self.scaler,
            "model": self.model,
            "error": self.error.state(),
            "effective_error": self.effective_error.state(),
            "samples_trained": self.samples_trained,
            "fit_count": self.fit_count,
            "last_fit_ts": self.last_fit_ts,
            # Keeping the buffer means a restart resumes with a warm model
            # instead of falling back to persistence for the first few minutes.
            "buf_X": [v.tolist() for v in self._buf_X],
            "buf_y": list(self._buf_y),
        }

    def load_state(self, state: dict) -> None:
        self.scaler = state.get("scaler")
        self.model = state.get("model")
        self.error = OnlineError.from_state(state.get("error", {}))
        self.effective_error = OnlineError.from_state(state.get("effective_error", {}))
        self.samples_trained = state.get("samples_trained", 0)
        self.fit_count = state.get("fit_count", 0)
        self.last_fit_ts = state.get("last_fit_ts")
        self._buf_X = deque(
            (np.asarray(v, dtype=float) for v in state.get("buf_X", [])), maxlen=self.window
        )
        self._buf_y = deque(state.get("buf_y", []), maxlen=self.window)


@dataclass
class _Pending:
    """A prediction awaiting the future sample that will grade it."""

    features: np.ndarray
    made_at_ts: float
    predictions: dict[str, float]  # emitted (shrunk) forecast
    raw_predictions: dict[str, float]  # unshrunk model output
    baseline: dict[str, float]  # value at prediction time = persistence forecast


@dataclass
class ForecastResult:
    target: str
    predicted: float | None
    horizon: int
    trained_on: int
    ready: bool
    mae: float | None
    naive_mae: float | None
    skill: float | None
    # How much of the emitted forecast comes from the model vs persistence.
    trust_weight: float = 0.0
    effective_skill: float | None = None
    fit_count: int = 0

    @property
    def source(self) -> str:
        """Plain-language description of what produced this number."""
        if self.predicted is None:
            return "warming up"
        if self.trust_weight <= 0.01:
            return "trend (model not beating baseline yet)"
        if self.trust_weight >= 0.99:
            return "learned model"
        return f"blend ({self.trust_weight*100:.0f}% model)"


class ForecastService:
    """Owns one :class:`OnlineForecaster` per target and the labelling loop."""

    TARGETS: tuple[str, ...] = ("cpu", "ram")

    def __init__(
        self,
        horizon: int,
        decay: float = 0.995,
        warmup: int = 120,
        sample_interval: float = 1.0,
        window: int = 2400,
        refit_every: int = 30,
        alpha: float = 100.0,
        min_fit: int = 120,
    ) -> None:
        self.horizon = horizon
        self.warmup = warmup
        self.sample_interval = sample_interval
        self.forecasters = {
            name: OnlineForecaster(
                name,
                horizon,
                window=window,
                refit_every=refit_every,
                alpha=alpha,
                min_fit=min_fit,
                decay=decay,
            )
            for name in self.TARGETS
        }
        # Keyed by the sample index the prediction is *for*.
        self._pending: dict[int, _Pending] = {}
        self._index = -1
        self.latest: dict[str, float | None] = {name: None for name in self.TARGETS}
        self.latest_weights: dict[str, float] = {}
        self.graded_rows: list[tuple] = []
        self.dropped_stale = 0

    # -- labelling -------------------------------------------------------
    def _resolve(self, index: int, actual: dict[str, float], now_ts: float) -> None:
        """Train on the prediction that targeted this sample, if any."""
        pending = self._pending.pop(index, None)
        if pending is None:
            return

        expected_gap = self.horizon * self.sample_interval
        actual_gap = now_ts - pending.made_at_ts
        # If the machine slept (or sampling stalled), the "future" we predicted
        # never happened on schedule. Training on it would teach nonsense.
        if actual_gap > expected_gap * 3 + 5:
            self.dropped_stale += 1
            return

        for target, forecaster in self.forecasters.items():
            truth = actual.get(target)
            if truth is None:
                continue
            emitted = pending.predictions.get(target)
            raw = pending.raw_predictions.get(target)
            # The anchor doubles as the persistence baseline: both are "the
            # value at the moment the prediction was made".
            baseline = pending.baseline.get(target)
            if baseline is None:
                continue

            forecaster.learn(pending.features, baseline, truth)

            naive_err = abs(truth - baseline)
            # Grade the RAW model against the baseline: that comparison is what
            # decides how much the model should be trusted next time.
            if raw is not None:
                forecaster.error.update(abs(truth - raw), naive_err)
            if emitted is not None:
                emitted_err = abs(truth - emitted)
                forecaster.effective_error.update(emitted_err, naive_err)
                self.graded_rows.append(
                    (
                        pending.made_at_ts,
                        target,
                        self.horizon,
                        emitted,
                        truth,
                        emitted_err,
                        naive_err,
                    )
                )

    def observe(self, features: np.ndarray, actual: dict[str, float], now_ts: float):
        """Advance one step: grade the due prediction, then forecast forward.

        Order matters. Grading first means the model is trained on the freshest
        available label before it is asked for a new prediction.
        """
        self._index += 1
        index = self._index

        self._resolve(index, actual, now_ts)

        predictions: dict[str, float | None] = {}
        raw_predictions: dict[str, float] = {}
        weights: dict[str, float] = {}
        for target, forecaster in self.forecasters.items():
            anchor = actual.get(target)
            if anchor is None:
                predictions[target] = None
                continue
            emitted, raw, weight = forecaster.predict_parts(features, anchor)
            predictions[target] = emitted
            weights[target] = weight
            if raw is not None:
                raw_predictions[target] = raw
        self.latest = predictions
        self.latest_weights = weights

        self._pending[index + self.horizon] = _Pending(
            features=features.copy(),
            made_at_ts=now_ts,
            predictions={k: v for k, v in predictions.items() if v is not None},
            raw_predictions=raw_predictions,
            baseline=dict(actual),
        )

        # Bound memory if resolution ever falls behind.
        if len(self._pending) > self.horizon * 10 + 50:
            for key in sorted(self._pending)[: len(self._pending) // 2]:
                self._pending.pop(key, None)
                self.dropped_stale += 1

        return self.results()

    def results(self) -> dict[str, ForecastResult]:
        out: dict[str, ForecastResult] = {}
        for target, forecaster in self.forecasters.items():
            out[target] = ForecastResult(
                target=target,
                predicted=self.latest.get(target),
                horizon=self.horizon,
                trained_on=forecaster.samples_trained,
                ready=forecaster.samples_trained >= self.warmup,
                mae=forecaster.error.model_mae,
                naive_mae=forecaster.error.naive_mae,
                skill=forecaster.error.skill,
                trust_weight=self.latest_weights.get(target, forecaster.trust_weight),
                effective_skill=forecaster.effective_error.skill,
                fit_count=forecaster.fit_count,
            )
        return out

    def drain_graded_rows(self) -> list[tuple]:
        rows, self.graded_rows = self.graded_rows, []
        return rows

    # -- persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": MODEL_FORMAT_VERSION,
            "n_features": N_FEATURES,
            "feature_names": list(FEATURE_NAMES),
            "horizon": self.horizon,
            "sample_interval": self.sample_interval,
            "saved_at": time.time(),
            "forecasters": {n: f.state() for n, f in self.forecasters.items()},
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(payload, tmp)
        tmp.replace(path)  # atomic: never leave a half-written model behind

    def load(self, path: Path) -> bool:
        """Restore a saved model. Returns False if it is stale/incompatible."""
        import joblib

        if not path.exists():
            return False
        try:
            payload = joblib.load(path)
        except Exception:
            return False
        if payload.get("format") != MODEL_FORMAT_VERSION:
            return False
        # A model trained on a different feature set would silently mispredict.
        if payload.get("n_features") != N_FEATURES:
            return False
        if payload.get("horizon") != self.horizon:
            return False
        try:
            for name, state in payload.get("forecasters", {}).items():
                if name in self.forecasters:
                    self.forecasters[name].load_state(state)
        except Exception:
            return False
        return True


# ---------------------------------------------------------------------------
# anomaly detection
# ---------------------------------------------------------------------------
@dataclass
class AnomalyResult:
    score: float  # 0..1, higher = more unusual
    tripped: list[tuple[str, float]] = field(default_factory=list)
    forest_ready: bool = False
    forest_score: float | None = None
    zscore_score: float | None = None
    window_size: int = 0
    last_trained: float | None = None
    train_count: int = 0

    @property
    def label(self) -> str:
        if self.score >= 0.85:
            return "severe"
        if self.score >= 0.65:
            return "notable"
        if self.score >= 0.4:
            return "mild"
        return "normal"


class AnomalyModel:
    """Layered anomaly scorer: instant robust z-scores + learned IsolationForest."""

    TRACKED = ("cpu", "ram", "swap", "disk_io_mbps", "net_io_mbps")
    # How many window rows to checkpoint to disk (see save()).
    PERSIST_WINDOW = 600

    def __init__(
        self,
        window: int = 1800,
        min_train: int = 180,
        retrain_every: int = 300,
        contamination: float = 0.02,
        z_threshold: float = 3.5,
    ) -> None:
        self.window_size = window
        self.min_train = min_train
        self.retrain_every = retrain_every
        self.contamination = contamination
        self.z_threshold = z_threshold

        self.trackers = {name: RobustZTracker() for name in self.TRACKED}
        self._window: deque[np.ndarray] = deque(maxlen=window)
        self._forest = None
        self._scaler = None
        self._since_retrain = 0
        self.train_count = 0
        self.last_trained: float | None = None
        # Calibration of decision_function -> 0..1, learned at each refit.
        self._df_median: float | None = None
        self._df_low: float | None = None

    # -- training --------------------------------------------------------
    def _retrain(self) -> None:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        data = np.asarray(self._window, dtype=float)
        if data.shape[0] < self.min_train:
            return

        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        forest = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=0,
            n_jobs=1,
        )
        forest.fit(scaled)

        # Calibrate on the training window itself: the median decision value is
        # "typical", the low percentile is "as odd as this machine usually gets".
        df = forest.decision_function(scaled)
        self._df_median = float(np.median(df))
        self._df_low = float(np.percentile(df, 1))
        if self._df_low >= self._df_median:
            self._df_low = self._df_median - 1e-3

        self._forest = forest
        self._scaler = scaler
        self._since_retrain = 0
        self.train_count += 1
        self.last_trained = time.time()

    def maybe_retrain(self, force: bool = False) -> bool:
        ready = len(self._window) >= self.min_train
        due = self._since_retrain >= self.retrain_every or self._forest is None
        if ready and (force or due):
            self._retrain()
            return True
        return False

    # -- scoring ---------------------------------------------------------
    def _forest_score(self, vector: np.ndarray) -> float | None:
        if self._forest is None or self._scaler is None:
            return None
        if self._df_median is None or self._df_low is None:
            return None
        try:
            scaled = self._scaler.transform(vector.reshape(1, -1))
            df = float(self._forest.decision_function(scaled)[0])
        except Exception:
            return None
        span = self._df_median - self._df_low
        if span <= 1e-9:
            return 0.0
        return float(np.clip((self._df_median - df) / span, 0.0, 1.0))

    def observe(self, vector: np.ndarray, metrics: dict[str, float]) -> AnomalyResult:
        """Update trackers, score the current point, and refit when due."""
        tripped: list[tuple[str, float]] = []
        max_abs_z = 0.0
        for name, tracker in self.trackers.items():
            value = metrics.get(name)
            if value is None:
                continue
            z = tracker.update(float(value))
            # Only unusually *high* usage is interesting; idle dips are not.
            if z >= self.z_threshold:
                tripped.append((name, float(z)))
            max_abs_z = max(max_abs_z, abs(z))

        # z == threshold maps to 0.5; z == 2*threshold saturates at 1.0.
        z_component = float(np.clip(max_abs_z / (2.0 * self.z_threshold), 0.0, 1.0))

        forest_component = self._forest_score(vector)

        self._window.append(np.asarray(vector, dtype=float))
        self._since_retrain += 1
        self.maybe_retrain()

        score = z_component if forest_component is None else max(z_component, forest_component)

        return AnomalyResult(
            score=float(score),
            tripped=sorted(tripped, key=lambda item: item[1], reverse=True),
            forest_ready=self._forest is not None,
            forest_score=forest_component,
            zscore_score=z_component,
            window_size=len(self._window),
            last_trained=self.last_trained,
            train_count=self.train_count,
        )

    def bulk_warm(self, vectors: Sequence[np.ndarray]) -> None:
        """Seed the rolling window from stored history, then fit immediately."""
        for vector in vectors:
            self._window.append(np.asarray(vector, dtype=float))
        self.maybe_retrain(force=True)

    # -- persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": MODEL_FORMAT_VERSION,
            "n_features": N_ANOMALY_FEATURES,
            "feature_names": list(ANOMALY_FEATURE_NAMES),
            "forest": self._forest,
            "scaler": self._scaler,
            "df_median": self._df_median,
            "df_low": self._df_low,
            "trackers": {n: t.state() for n, t in self.trackers.items()},
            "train_count": self.train_count,
            "last_trained": self.last_trained,
            # Only a slice of the window is checkpointed, as float32. The point
            # is to avoid a cold start, not to reproduce the buffer exactly -
            # live sampling refills it within minutes, and this keeps the
            # periodic save at tens of KB instead of hundreds.
            "window": np.asarray(
                list(self._window)[-self.PERSIST_WINDOW :], dtype=np.float32
            ),
            "saved_at": time.time(),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(payload, tmp)
        tmp.replace(path)

    def load(self, path: Path) -> bool:
        import joblib

        if not path.exists():
            return False
        try:
            payload = joblib.load(path)
        except Exception:
            return False
        if payload.get("format") != MODEL_FORMAT_VERSION:
            return False
        if payload.get("n_features") != N_ANOMALY_FEATURES:
            return False
        try:
            self._forest = payload.get("forest")
            self._scaler = payload.get("scaler")
            self._df_median = payload.get("df_median")
            self._df_low = payload.get("df_low")
            self.train_count = payload.get("train_count", 0)
            self.last_trained = payload.get("last_trained")
            for name, state in payload.get("trackers", {}).items():
                if name in self.trackers:
                    self.trackers[name] = RobustZTracker.from_state(state)
            window = payload.get("window")
            if window is not None:
                for row in np.asarray(window, dtype=float):
                    self._window.append(np.asarray(row, dtype=float))
        except Exception:
            return False
        return True


# ---------------------------------------------------------------------------
# health score
# ---------------------------------------------------------------------------
def health_score(
    cpu: float,
    ram: float,
    disk: float,
    swap: float = 0.0,
    temp_c: float | None = None,
    weights: dict[str, float] | None = None,
) -> int:
    """0-100 headroom score, where 100 means 'nothing is under pressure'.

    Improvement over the original formula: disk *capacity* is nearly constant
    and used to dominate the score, so pressure is now measured against the
    point where each resource actually starts hurting, not raw percentages.
    """
    weights = weights or {
        "cpu": 0.30,
        "ram": 0.30,
        "disk": 0.15,
        "swap": 0.15,
        "thermal": 0.10,
    }

    def pressure(value: float, soft: float, hard: float) -> float:
        """0 below `soft`, ramping to 1 at `hard`."""
        if value <= soft:
            return 0.0
        if value >= hard:
            return 1.0
        return (value - soft) / (hard - soft)

    components = {
        "cpu": pressure(cpu, 40.0, 95.0),
        "ram": pressure(ram, 55.0, 95.0),
        # Only near-full disks matter; 60% used is harmless.
        "disk": pressure(disk, 80.0, 98.0),
        # Any sustained swapping is a real slowdown, so it ramps early.
        "swap": pressure(swap, 5.0, 60.0),
    }
    total_weight = weights["cpu"] + weights["ram"] + weights["disk"] + weights["swap"]
    penalty = sum(components[k] * weights[k] for k in components)

    if temp_c is not None:
        components["thermal"] = pressure(temp_c, 70.0, 95.0)
        penalty += components["thermal"] * weights["thermal"]
        total_weight += weights["thermal"]

    if total_weight <= 0:
        return 100
    normalized = penalty / total_weight
    return int(round(max(0.0, min(100.0, 100.0 * (1.0 - normalized)))))
