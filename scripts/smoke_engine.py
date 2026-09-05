"""End-to-end check of the live engine: sampling cadence, storage, feedback."""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurotrack.config import Config
from neurotrack.engine import MonitorEngine

# Long enough to cross a minute boundary (exercises rollups) and to let the
# forecaster reach its first fit.
RUN_SECONDS = 75


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="neurotrack-smoke-"))
    cfg = Config(data_dir=tmp)
    cfg.sample_interval = 1.0
    cfg.maintenance_every = 10          # exercise rollups/pruning quickly
    cfg.flush_every = 5
    cfg.anomaly_min_train = 15
    cfg.anomaly_retrain_every = 10
    cfg.forecast_horizon = 5
    cfg.model_warmup_samples = 10
    cfg.forecast_refit_every = 5
    cfg.forecast_min_fit = 20           # so a short run reaches the first fit
    cfg.ensure_dirs()

    engine = MonitorEngine(cfg)
    engine.start()
    print(f"engine started (data dir {tmp})")

    ok = True
    t0 = time.time()
    while time.time() - t0 < RUN_SECONDS:
        time.sleep(5)
        snap = engine.snapshot()
        status = engine.status()
        if snap is None:
            print("  ... no snapshot yet")
            continue
        s = snap.sample
        fc = snap.forecasts.get("cpu")
        print(
            f"  t={time.time()-t0:5.1f}s tick={status['tick']:3d} "
            f"cpu={s.cpu:5.1f} ram={s.ram:5.1f} health={snap.health:3d} "
            f"anom={snap.anomaly.score:.2f} age={status['snapshot_age']:.2f}s "
            f"alerts={len(snap.feedback.alerts)} advice={len(snap.feedback.advice)}"
        )
        if fc is not None:
            print(
                f"        forecast cpu: {fc.predicted} source={fc.source!r} "
                f"pairs={fc.trained_on} refits={fc.fit_count}"
            )

    status = engine.status()
    print("\nstatus:")
    for key, value in status.items():
        print(f"  {key}: {value}")

    # --- cadence check ---------------------------------------------------
    expected = RUN_SECONDS / cfg.sample_interval
    if not (expected * 0.8 <= status["tick"] <= expected * 1.2):
        ok = False
        print(f"!! FAIL: expected ~{expected:.0f} ticks, got {status['tick']}")
    else:
        print(f"\ncadence OK: {status['tick']} ticks in {RUN_SECONDS}s")

    if status["errors"]:
        ok = False
        print(f"!! FAIL: {status['errors']} errors, last={status['last_error']}")

    # --- snapshot freshness ---------------------------------------------
    if status["snapshot_age"] is None or status["snapshot_age"] > 3:
        ok = False
        print(f"!! FAIL: snapshot is stale ({status['snapshot_age']}s)")

    engine.stop()
    print("engine stopped")

    # --- storage check ---------------------------------------------------
    engine.storage.build_rollups()
    count = engine.storage.sample_count()
    rollups = engine.storage.rollups_since(0)
    events = engine.storage.recent_events(limit=20)
    preds = engine.storage.prediction_scores("cpu", limit=100)
    print(f"\nstored: {count} raw samples, {len(rollups)} rollups, "
          f"{len(events)} events, {len(preds)} graded predictions")
    print(f"db size: {engine.storage.database_size_mb():.3f} MB")

    if count < expected * 0.7:
        ok = False
        print(f"!! FAIL: only {count} rows persisted")
    if not preds:
        ok = False
        print("!! FAIL: no graded predictions were written")

    # models were checkpointed
    for path in (cfg.forecaster_path, cfg.anomaly_path):
        if path.exists():
            print(f"model saved: {path.name} ({path.stat().st_size/1024:.1f} KB)")
        else:
            ok = False
            print(f"!! FAIL: {path} missing")

    # --- restart check: models + history reload --------------------------
    engine2 = MonitorEngine(cfg)
    print(
        f"\nrestart: models_restored={engine2.models_restored} "
        f"history_reloaded={len(engine2.history())}"
    )
    if not engine2.models_restored["forecaster"]:
        ok = False
        print("!! FAIL: forecaster did not restore")
    if len(engine2.history()) == 0:
        ok = False
        print("!! FAIL: history did not reload")
    engine2.storage.close()

    if len(sys.argv) > 1 and sys.argv[1] == "--keep":
        print(f"\nkept data dir: {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
