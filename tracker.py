"""NeuroTrack AI - Streamlit entry point.

Run with:
    streamlit run tracker.py

The application itself lives in the ``neurotrack`` package:

    neurotrack/collector.py  hardware sampling (psutil, cross-platform)
    neurotrack/storage.py    SQLite persistence, rollups, retention
    neurotrack/features.py   feature engineering shared by both models
    neurotrack/models.py     forecaster + anomaly detector
    neurotrack/feedback.py   thresholds, hysteresis, actionable advice
    neurotrack/sessions.py   activity session segmentation
    neurotrack/engine.py     the 1 Hz background sampler thread
    neurotrack/ui.py         this dashboard
"""

from neurotrack.ui import main

main()
