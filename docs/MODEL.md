# How the models are trained and used

Everything here runs on your machine, learns from your machine, and stays on your
machine. There are no pretrained weights, no downloads, and nothing is uploaded.
On a fresh install the models know nothing; they become useful after a few
minutes of sampling.

Two models run side by side, and they solve different problems:

| | Forecaster | Anomaly detector |
|---|---|---|
| Question | "what will CPU/RAM be in *H* seconds?" | "is right now unusual for this machine?" |
| Type | supervised regression | unsupervised |
| Where labels come from | waiting *H* seconds | none needed |
| Algorithm | Ridge, refit on a rolling window | robust z-scores + IsolationForest |
| Usable after | ~2-3 minutes | ~30 seconds |

---

## 1. The forecaster

### The labelling trick

Forecasting normally needs labelled data, which normally means a human. Here the
label arrives on its own, you just have to wait for it.

```
t = 0s    build features from history, predict CPU at t = 15s   ->  "62%"
          store (features, prediction, current_value) in a queue
...
t = 15s   the real value is now known: 71%
          -> grade the old prediction (error = 9)
          -> train on (features_from_t0, label = 71 - 62_current)
```

Every second produces exactly one new training example, delayed by the forecast
horizon. This is sometimes called *delayed-label* or *self-supervised* learning.
`ForecastService.observe()` in [`neurotrack/models.py`](../neurotrack/models.py)
implements it, and the order inside one tick matters:

1. **grade + train** on the label that just became available;
2. **then predict** forward.

Doing it in that order means each prediction is made by a model that has seen
every label available at that moment, and never a label from the future.

### It predicts the change, not the value

The training label is `y[t+H] - y[t]`, and the emitted forecast is
`y[t] + predicted_change`.

This is the single most important design decision, because of what the baseline
is. The naive forecast — "in 15 seconds it will be whatever it is now", called
*persistence* — is a genuinely strong opponent for smooth signals. Persistence is
exactly `change = 0`. So a model with all-zero weights **equals** the baseline
rather than having to rediscover it, and any learning starts from parity instead
of from a deficit. It also keeps the target centred near zero and roughly
stationary, which is much easier to fit than absolute CPU%.

### Why Ridge on a rolling window, and not online SGD

The first implementation used `SGDRegressor.partial_fit` — one gradient step per
sample, the textbook "online learning" answer. **Measured against the baseline it
lost.** Consecutive samples one second apart are nearly identical, so each
gradient step is dominated by noise, and every learning rate either crawled or
oscillated. Averaged SGD (`average=True`) was worse still, diverging to a skill
of about **-1.7**.

What works instead: keep a buffer of the last `forecast_window` labelled pairs
and refit a closed-form Ridge every `forecast_refit_every` samples. No learning
rate to tune, far more stable, and it costs a few milliseconds every 30 seconds.
It is still incremental and still adapts — the window slides, so old behaviour
ages out.

The sweep scripts that produced this conclusion are kept in
[`scripts/`](../scripts) precisely so the reasoning is auditable:
`sweep.py`, `sweep_ridge.py`, `sweep_horizon.py`.

### Features

48 features, built by `build_forecast_features()` in
[`neurotrack/features.py`](../neurotrack/features.py):

- current CPU, hottest core, RAM, swap, disk I/O, network I/O, load, process count
- lagged CPU and RAM at 1, 2, 3, 5, 8, 15, 30, 60 seconds back
- rolling mean and standard deviation over 5, 15 and 60 second windows
- deltas and deviation from the recent local mean (momentum)
- hour-of-day and day-of-week as sine/cosine pairs, so 23:59 sits next to 00:00
- a constant bias term

**Deliberately excluded: uptime and disk-percent-used.** Both drift
monotonically. Standardising a monotonically increasing feature turns it into a
proxy for "time index", which a linear model will happily fit and then
extrapolate nonsense from. Including uptime measurably degraded accuracy and gave
it one of the largest weights in the model — a textbook spurious feature.

### Honest accuracy, and the shrinkage safety net

Some signals simply are not forecastable, and the app says so instead of
pretending otherwise.

Every graded prediction updates two error trackers: the model's error and the
persistence baseline's error over the same predictions. From those comes **skill**:

```
skill = 1 - (model_MAE / baseline_MAE)
```

- `skill > 0` — the model beats "nothing will change"
- `skill <= 0` — it does not, and is adding nothing

Measured behaviour, with the honest label delay applied:

| horizon | CPU skill | RAM skill |
|---|---|---|
| 3s | ~0% | **+21%** |
| 5s | ~0% | **+20%** |
| 15s | ~0% | **+13%** |
| 30s | ~+2% | **+11%** |
| 60s | ~0% | **+13%** |

**RAM is genuinely forecastable. Second-scale CPU essentially is not.** That is a
real property of the signals, not a bug: RAM moves in trends (allocations, leaks,
caches filling), while CPU at one-second resolution behaves close to a random
walk around its local level.

So the forecast is shrunk toward persistence based on measured skill:

```
emitted_forecast = current_value + trust_weight * predicted_change

trust_weight = 0                          if skill <= 0, or < 60 graded predictions
             = clamp(skill / 0.10, 0, 1)  otherwise
```

A metric the model cannot predict collapses to `trust_weight = 0`, and the
forecast quietly becomes the baseline instead of producing confident nonsense.
A metric it can predict gets the full benefit. The **Model** page shows the raw
model skill, the trust weight, and the skill of what is actually emitted, and the
**Live** page labels each forecast with its source (`learned model`, `blend (60%
model)`, or `trend (model not beating baseline yet)`).

Every graded prediction is also written to a `predictions` table with the value
that actually occurred, so the accuracy claims are auditable rather than
asserted — see the audit trail chart on the Model page.

### Sleep and suspend

A laptop that suspends breaks the assumption that the label arrives *H* seconds
later. On resume, a pending prediction's "future" never happened on schedule, so
training on it would teach the model nonsense. Labels whose actual gap exceeds
`3H + 5` seconds are discarded, and the count is reported as
`stale_labels_dropped`.

---

## 2. The anomaly detector

Unsupervised, and layered because the two layers fail in different ways.

### Layer 1 — streaming robust z-scores

Per metric (CPU, RAM, swap, disk I/O, network I/O), an exponentially weighted
mean and mean-absolute-deviation are updated every sample:

```
z = (value - ewma_mean) / (1.4826 * ewma_mad)
```

MAD rather than standard deviation is the point. One enormous spike inflates a
standard deviation enough to hide everything after it; MAD is far less affected.
The `1.4826` factor rescales MAD to be comparable to a standard deviation for
normally distributed data. This layer needs about 30 samples, so it works almost
immediately, but it only ever looks at one metric at a time.

### Layer 2 — IsolationForest on a rolling window

`IsolationForest` is refit on the last `anomaly_window` samples (default 1800, so
30 minutes) every `anomaly_retrain_every` samples. It sees all metrics together,
which is what catches abnormal *combinations* — moderate CPU that would pass a
per-metric threshold, but occurring with unusual disk and network activity at the
same time.

Raw `decision_function` output is not a probability and its range depends on the
data, so it is calibrated at each refit: the median score on the training window
maps to 0, the 1st percentile maps to 1, linearly, clipped.

The final score is `max(z_layer, forest_layer)`, in 0-1, bucketed as
`normal` / `mild` / `notable` / `severe`.

---

## 3. From score to advice

Raw model output is not feedback. [`neurotrack/feedback.py`](../neurotrack/feedback.py)
converts it, with three properties that make the difference between useful and
infuriating at a 1 Hz update rate:

- **Hysteresis** — a rule fires only after the condition holds for
  `trigger_samples` consecutive samples, and clears at a *lower* threshold than
  it triggers at. A plain `cpu > 85` check at 1 Hz would fire dozens of times a
  minute while hovering near the line.
- **Cooldown** — the same alert is not re-announced within `alert_cooldown`
  seconds.
- **Attribution** — advice names the responsible processes. "CPU is high" is not
  actionable; "chrome (240%), node (95%) are the top consumers" is.

Forecast-driven warnings are only raised when `trust_weight > 0.05`. If the model
is not beating the baseline, its "forecast" is just the current value, and
warning about it would be warning about nothing new.

### Trend projections are separate, and deliberately simpler

Time-to-exhaustion for RAM and disk uses a plain least-squares fit over the
recent window, not the ML model. This is the memory-leak detector: a steady
upward slope in RAM is exactly the signature of a process that never frees
memory. A straight line is the right tool — it is explainable, needs no training,
and works over minutes-to-hours where the second-scale model has nothing to say.

---

## 4. Cost and persistence

The full per-sample chain — collect, build features, grade, retrain, predict,
score, evaluate feedback — measures about **5 ms**, roughly **0.5% of one core**
at 1 Hz. Per-process scanning is the expensive part and runs every
`process_scan_every` samples instead of every sample.

Models are checkpointed with `joblib` every `model_save_interval` seconds
(default 300) and on clean shutdown, then reloaded at startup, so learning
survives restarts. Checkpoints are written atomically via a temporary file, so an
interrupted save cannot leave a corrupt model. A saved model is **rejected** on
load if the format version, feature count or horizon does not match the current
code — a model trained on a different feature set would otherwise silently
mispredict.

## 5. Auditing it yourself

```bash
python train_model.py                        # replay stored history, report skill
python train_model.py --horizons 3,5,15,30   # find what is predictable for you
python train_model.py --save                 # train on history and checkpoint
```

`train_model.py` replays your stored samples through the same `ForecastService`
the live app uses, with the same label delay, so its numbers are directly
comparable to what the dashboard reports.
