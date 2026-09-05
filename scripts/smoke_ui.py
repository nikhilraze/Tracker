"""Render every page headlessly via streamlit.testing and assert no exceptions.

Serving the HTML shell proves nothing: Streamlit only executes the script when a
session connects. AppTest drives a real script run, so template errors, bad
column configs and API misuse surface here.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="neurotrack-ui-"))
    # Point the app at a throwaway data dir and make models converge fast.
    import os

    os.environ["NEUROTRACK_DATA_DIR"] = str(tmp)
    os.environ["NEUROTRACK_FORECAST_MIN_FIT"] = "15"
    os.environ["NEUROTRACK_FORECAST_REFIT_EVERY"] = "5"
    os.environ["NEUROTRACK_FORECAST_HORIZON"] = "3"
    os.environ["NEUROTRACK_ANOMALY_MIN_TRAIN"] = "15"
    os.environ["NEUROTRACK_ANOMALY_RETRAIN_EVERY"] = "10"
    os.environ["NEUROTRACK_MODEL_WARMUP_SAMPLES"] = "10"

    from streamlit.testing.v1 import AppTest

    from neurotrack.ui import PAGES

    ok = True
    pages = list(PAGES)

    # Warm the engine so pages have real data (forecasts, processes, anomaly).
    from neurotrack.engine import get_engine

    engine = get_engine()
    print("warming engine for 30s so every page has real data to render...")
    time.sleep(30)
    snap = engine.snapshot()
    print(
        f"  tick={engine.status()['tick']} "
        f"forecast_cpu={snap.forecasts['cpu'].predicted} "
        f"processes={len(snap.processes)} anomaly={snap.anomaly.score:.2f}"
    )
    engine.storage.build_rollups()

    for page in pages:
        app = AppTest.from_file(str(ROOT / "tracker.py"), default_timeout=90)
        app.run()
        if app.exception:
            ok = False
            print(f"[{page}] EXCEPTION on initial run:")
            for exc in app.exception:
                print("   ", exc.value)
            continue

        # Select the page in the sidebar radio.
        try:
            app.sidebar.radio[0].set_value(page).run()
        except Exception as exc:
            ok = False
            print(f"[{page}] failed to select page: {exc}")
            continue

        if app.exception:
            ok = False
            print(f"[{page}] EXCEPTION after selecting page:")
            for exc in app.exception:
                print("   ", exc.value)
        else:
            widgets = (
                len(app.metric) + len(app.dataframe) + len(app.markdown) + len(app.caption)
            )
            print(f"[{page}] OK - {len(app.metric)} metrics, {widgets} elements rendered")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
