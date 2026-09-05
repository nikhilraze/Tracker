# 🧠 NeuroTrack AI — per-second system tracking with real-time feedback

Samples your machine **every second**, learns what normal looks like **on that
machine**, and tells you what to do about it — with the honesty to say when its
own prediction isn't worth trusting.

Everything stays local. No account, no telemetry, no cloud.

---

## What it does

**Per-second collection.** A background thread samples at 1 Hz: CPU (total and
hottest core), RAM, swap, disk usage, disk and network throughput in MB/s, load
average, process count, CPU frequency, battery, temperature and fan speed.
Sampling is decoupled from the UI, so it keeps running with the browser closed.

**Process attribution.** Live ranking of which processes are actually consuming
CPU and memory, so advice can name the culprit instead of just reporting that
something is wrong.

**Learned real-time feedback.**
- CPU/RAM forecast for the next N seconds, trained continuously on your own data
- anomaly detection that catches unusual *combinations* of metrics, not just
  single thresholds
- alerts with hysteresis and cooldown, so a metric hovering near a threshold
  doesn't spam you
- a memory-leak detector: time-to-full projection from the recent RAM trend
- actionable advice naming the responsible processes

**Usage habits.** Activity sessions, screen time per day, load by hour of day, and
optional break reminders.

**History that scales.** SQLite storage with automatic per-minute rollups and
tiered retention — raw per-second rows for hours, per-minute aggregates for
months. Per-second sampling produces 86,400 rows a day, which is why the original
CSV approach had to go.

**An honest model page.** Live accuracy against a naive baseline, the features the
model is keying on, and an audit trail of past predictions next to what actually
happened.

---

## Install and run

```bash
git clone https://github.com/nikhilraze/Tracker.git
cd Tracker
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run tracker.py
```

Requires Python 3.10+. Works on Linux, macOS and Windows; features that depend on
hardware (battery, temperature, fan) appear only where the OS exposes them.

Data is written to `~/.neurotrack/` — set `NEUROTRACK_DATA_DIR` to change it.

---

## Pages

| Page | What's there |
|---|---|
| **Live** | 1 Hz metrics, short-term forecast, anomaly score, what to do now |
| **Processes** | live CPU/memory ranking per process |
| **Analytics** | per-second detail, per-minute long range, usage habits |
| **Insights** | active alerts, recommended actions, projections, event history |
| **Model** | training state, accuracy vs baseline, learned weights, prediction audit |
| **Settings** | diagnostics, storage stats, CSV export, full configuration |

---

## How the AI works

Short version: two models, both trained from scratch on your machine.

1. **Forecaster** — predicts CPU and RAM a few seconds ahead. It gets its labels
   for free by waiting: a prediction made now is graded against reality when that
   moment arrives, and the pair becomes a training example. It learns the
   *change* rather than the absolute value, so it starts at parity with the naive
   "nothing will change" baseline instead of behind it.

2. **Anomaly detector** — streaming robust z-scores per metric for instant
   coverage, plus an `IsolationForest` refit on a rolling window to catch odd
   combinations across metrics.

The part worth knowing: **the app measures its own forecast against that naive
baseline and shrinks toward the baseline when it isn't winning.** Measured on
real traces, RAM is genuinely forecastable (about +13% to +21% better than the
baseline) while second-scale CPU essentially is not — CPU at one-second
resolution behaves close to a random walk. Rather than hide that, the forecast
falls back to the baseline and the UI labels the source. You get the gain where
it exists and no confident nonsense where it doesn't.

**→ [docs/MODEL.md](docs/MODEL.md)** covers the training loop, the feature set,
what was measured, and the approaches that were tried and rejected (per-sample
SGD lost to the baseline; averaged SGD diverged; uptime as a feature was actively
harmful).

### Audit or pre-train it yourself

```bash
python train_model.py                        # replay stored history, report skill
python train_model.py --horizons 3,5,15,30   # find what's predictable on your machine
python train_model.py --save                 # train on history so a fresh start is warm
```

---

## Configuration

Every setting is overridable with a `NEUROTRACK_`-prefixed environment variable
(see the Settings page for the full list with current values):

```bash
NEUROTRACK_SAMPLE_INTERVAL=1.0        # seconds between samples
NEUROTRACK_FORECAST_HORIZON=15        # how far ahead to predict
NEUROTRACK_RAW_RETENTION_HOURS=12     # how long to keep per-second rows
NEUROTRACK_CPU_HIGH=85                # alert thresholds
NEUROTRACK_BREAK_REMINDER_MINUTES=55
NEUROTRACK_DATA_DIR=~/.neurotrack
```

---

## Project layout

```
tracker.py              Streamlit entry point
train_model.py          offline training / evaluation CLI
neurotrack/
  collector.py          psutil sampling, cross-platform, per-process attribution
  storage.py            SQLite, rollups, tiered retention
  features.py           feature engineering shared by both models
  models.py             forecaster, anomaly detector, health score
  feedback.py           thresholds, hysteresis, actionable advice
  sessions.py           activity session segmentation
  engine.py             the 1 Hz background sampler thread
  ui.py                 the dashboard
scripts/                smoke tests and the hyperparameter sweeps behind the defaults
docs/MODEL.md           how the models are trained and used
```

## Verifying a change

```bash
python scripts/smoke_models.py    # models beat / safely match the baseline
python scripts/smoke_engine.py    # 1 Hz cadence, storage, restart recovery
python scripts/smoke_ui.py        # every page renders without exceptions
```

---

## Notes and limits

- **Activity is inferred from resource usage, not input.** Reading real
  keyboard/mouse idle time needs OS-specific APIs this project avoids, so a long
  build with nobody at the desk counts as active. The UI says "activity", never
  "presence".
- **Trend projections are straight lines.** Time-to-full for RAM and disk is a
  least-squares extrapolation — good for catching leaks, not a guarantee.
- **A process can show more than 100% CPU** because it's measured across all
  cores; the Processes page also shows the normalised share.
- **In a container**, battery and thermal sensors are usually absent and process
  visibility is limited to the container's namespace.

## Author

**Nikhil Raj** — if this is useful, a ⭐ on GitHub is appreciated.
