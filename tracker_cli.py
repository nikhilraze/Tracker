"""Console entry point: ``tracker`` launches the Streamlit dashboard.

Any extra arguments are forwarded to Streamlit, so this works:

    tracker --server.port 8600 --server.headless true
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    app_path = Path(__file__).with_name("tracker.py")
    if not app_path.exists():
        print(f"cannot find {app_path}", file=sys.stderr)
        return 1

    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    if argv:
        # Streamlit needs `--` before script args, but flags for Streamlit
        # itself go straight through.
        command.extend(argv)

    try:
        return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
