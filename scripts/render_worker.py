from pathlib import Path
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]

LOCAL_LIVE = ROOT / "data" / "live"
PERSISTENT_LIVE = Path(
    os.environ.get(
        "V5_LIVE_DATA_DIR",
        "/var/data",
    )
)

SEED_ARCHIVE = ROOT / "seed" / "live_seed.tar.gz"
SEED_MARKER = PERSISTENT_LIVE / ".v5_seed_complete"

PROCESSED_DIR = ROOT / "data" / "processed"
PERSISTENT_PROCESSED = PERSISTENT_LIVE / "processed"
PROCESSED_SEED_ARCHIVE = ROOT / "seed" / "processed_seed.tar.gz"
PROCESSED_SEED_MARKER = PERSISTENT_PROCESSED / ".v5_processed_seed_complete"

INTERVAL_SECONDS = 1 * 60 * 60


def prepare_live_storage():

    # Local Mac development continues using the normal
    # repository data/live directory.
    if not os.environ.get("RENDER"):
        print(
            f"Local mode: using {LOCAL_LIVE}",
            flush=True,
        )
        LOCAL_LIVE.mkdir(
            parents=True,
            exist_ok=True,
        )
        return

    print(
        f"Render mode: persistent live storage = "
        f"{PERSISTENT_LIVE}",
        flush=True,
    )

    PERSISTENT_LIVE.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Seed the persistent disk only once.
    if not SEED_MARKER.exists():

        if SEED_ARCHIVE.exists():

            print(
                "Initializing persistent V5 state "
                "from live seed...",
                flush=True,
            )

            with tarfile.open(
                SEED_ARCHIVE,
                "r:gz",
            ) as tar:
                tar.extractall(
                    PERSISTENT_LIVE
                )

            SEED_MARKER.write_text(
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            print(
                "Persistent V5 state seeded.",
                flush=True,
            )

        else:
            print(
                "WARNING: No seed archive found. "
                "Starting persistent live directory empty.",
                flush=True,
            )

    else:
        print(
            "Existing persistent V5 state found. "
            "Seed will NOT be reapplied.",
            flush=True,
        )

    # Replace data/live with a symlink to Render's disk.
    LOCAL_LIVE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if LOCAL_LIVE.is_symlink():
        LOCAL_LIVE.unlink()

    elif LOCAL_LIVE.exists():

        if LOCAL_LIVE.is_dir():
            shutil.rmtree(
                LOCAL_LIVE
            )
        else:
            LOCAL_LIVE.unlink()

    LOCAL_LIVE.symlink_to(
        PERSISTENT_LIVE,
        target_is_directory=True,
    )

    print(
        f"{LOCAL_LIVE} -> {PERSISTENT_LIVE}",
        flush=True,
    )

    # ---------------------------------------------------------
    # Persistent processed model inputs
    # ---------------------------------------------------------

    PERSISTENT_PROCESSED.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Processed model inputs are static deployment assets.
    # Refresh them from the current archive on every Render
    # process start so newly added dependencies are available.
    if not PROCESSED_SEED_ARCHIVE.exists():
        raise FileNotFoundError(
            f"Missing processed seed archive: "
            f"{PROCESSED_SEED_ARCHIVE}"
        )

    print(
        "Refreshing persistent processed model data "
        "from deployment seed...",
        flush=True,
    )

    with tarfile.open(
        PROCESSED_SEED_ARCHIVE,
        "r:gz",
    ) as tar:
        tar.extractall(
            PERSISTENT_PROCESSED
        )

    PROCESSED_SEED_MARKER.write_text(
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    print(
        "Persistent processed model data refreshed.",
        flush=True,
    )

    if PROCESSED_DIR.is_symlink():
        PROCESSED_DIR.unlink()

    elif PROCESSED_DIR.exists():
        if PROCESSED_DIR.is_dir():
            shutil.rmtree(PROCESSED_DIR)
        else:
            PROCESSED_DIR.unlink()

    PROCESSED_DIR.symlink_to(
        PERSISTENT_PROCESSED,
        target_is_directory=True,
    )

    print(
        f"{PROCESSED_DIR} -> {PERSISTENT_PROCESSED}",
        flush=True,
    )


def run_command(args, label):

    print(
        "\n" + "=" * 100,
        flush=True,
    )
    print(
        label,
        datetime.now(
            timezone.utc
        ).isoformat(),
        flush=True,
    )
    print(
        "=" * 100,
        flush=True,
    )

    result = subprocess.run(
        args,
        cwd=ROOT,
    )

    return result.returncode


def run_pipeline():

    rc = run_command(
        [
            sys.executable,
            "scripts/run_live_v5.py",
        ],
        "RUN V5 LIVE PIPELINE",
    )

    if rc != 0:
        print(
            f"V5 pipeline failed with exit code {rc}",
            flush=True,
        )
        return

    rc = run_command(
        [
            sys.executable,
            "scripts/push_dashboard.py",
        ],
        "PUSH V5 DASHBOARD",
    )

    if rc != 0:
        print(
            f"Dashboard push failed with exit code {rc}",
            flush=True,
        )
        return

    print(
        "V5 pipeline + dashboard push complete.",
        flush=True,
    )


def main():

    prepare_live_storage()

    while True:

        try:
            run_pipeline()

        except Exception as exc:
            print(
                f"Worker error: {exc}",
                flush=True,
            )

        print(
            f"Sleeping "
            f"{INTERVAL_SECONDS // 3600} hours...",
            flush=True,
        )

        time.sleep(
            INTERVAL_SECONDS
        )


if __name__ == "__main__":
    main()
