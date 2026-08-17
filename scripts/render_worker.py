import subprocess
import sys
import time
from datetime import datetime, timezone

INTERVAL_SECONDS = 6 * 60 * 60  # every 6 hours

def run_pipeline():
    print("\n" + "=" * 100)
    print(
        "RENDER V5 RUN:",
        datetime.now(timezone.utc).isoformat()
    )
    print("=" * 100, flush=True)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_live_v5.py",
        ]
    )

    if result.returncode != 0:
        print(
            f"V5 pipeline failed with exit code "
            f"{result.returncode}",
            flush=True,
        )
        return

    result = subprocess.run(
        [
            sys.executable,
            "scripts/push_dashboard.py",
        ]
    )

    if result.returncode != 0:
        print(
            f"Dashboard push failed with exit code "
            f"{result.returncode}",
            flush=True,
        )
        return

    print(
        "V5 pipeline + dashboard push complete.",
        flush=True,
    )

while True:
    try:
        run_pipeline()
    except Exception as exc:
        print(
            f"Worker error: {exc}",
            flush=True,
        )

    print(
        f"Sleeping {INTERVAL_SECONDS // 3600} hours...",
        flush=True,
    )

    time.sleep(INTERVAL_SECONDS)
