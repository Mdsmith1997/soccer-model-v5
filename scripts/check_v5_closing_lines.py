from pathlib import Path
from datetime import datetime, timezone
import subprocess
import sys
import os

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

LEDGER_FILE = (
    ROOT
    / "data"
    / "live"
    / "v5_live_bet_ledger.csv"
)

HISTORY_FILE = (
    ROOT
    / "data"
    / "live"
    / "odds_h2h_history.csv"
)

ODDS_FETCHER = (
    ROOT
    / "scripts"
    / "fetch_us_soccer_odds.py"
)

CLV_REPORT = (
    ROOT
    / "scripts"
    / "build_live_v5_clv_report.py"
)

ENV_FILE = (
    ROOT
    / ".private"
    / "odds_api.env"
)


def load_private_environment():

    if not ENV_FILE.exists():
        return

    for raw_line in ENV_FILE.read_text().splitlines():

        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)

        key = key.strip()
        value = value.strip()

        if key and value:
            os.environ.setdefault(
                key,
                value,
            )


# ============================================================
# SETTINGS
# ============================================================

# We want two late-market observations:
#
# T-60 = useful late-market checkpoint
# T-15 = preferred closing-line observation
#
# A checkpoint is considered already captured when there is
# a snapshot inside its target band.
CHECKPOINTS = [
    {
        "name": "T-60",
        "start": 60,
        "end": 31,
    },
    {
        "name": "T-15",
        "start": 30,
        "end": 0,
    },
]


def utc_now():
    return pd.Timestamp(
        datetime.now(timezone.utc)
    )


def load_history():

    if not HISTORY_FILE.exists():
        return pd.DataFrame()

    history = pd.read_csv(
        HISTORY_FILE,
        low_memory=False,
    )

    if history.empty:
        return history

    history["snapshot_time"] = pd.to_datetime(
        history["snapshot_time"],
        utc=True,
        errors="coerce",
    )

    history["commence_time"] = pd.to_datetime(
        history["commence_time"],
        utc=True,
        errors="coerce",
    )

    history["event_id"] = (
        history["event_id"]
        .astype(str)
        .str.strip()
    )

    return history


def checkpoint_already_captured(
    history,
    event_id,
    kickoff,
    start_minutes,
    end_minutes,
):

    if history.empty:
        return False

    x = history[
        history["event_id"].eq(
            str(event_id).strip()
        )
    ].copy()

    if x.empty:
        return False

    x = x[
        x["snapshot_time"].notna()
    ].copy()

    if x.empty:
        return False

    x["minutes_before_kickoff"] = (
        (
            kickoff
            -
            x["snapshot_time"]
        )
        .dt.total_seconds()
        /
        60.0
    )

    captured = x[
        (
            x["minutes_before_kickoff"]
            <= start_minutes
        )
        &
        (
            x["minutes_before_kickoff"]
            >
            end_minutes
        )
    ]

    return not captured.empty


def main():

    load_private_environment()

    print()
    print("=" * 100)
    print("V5 CLOSING-LINE CHECK")
    print("=" * 100)
    print()

    if not LEDGER_FILE.exists():
        raise RuntimeError(
            f"Ledger not found: {LEDGER_FILE}"
        )

    ledger = pd.read_csv(
        LEDGER_FILE,
        low_memory=False,
    )

    ledger["commence_time"] = pd.to_datetime(
        ledger["commence_time"],
        utc=True,
        errors="coerce",
    )

    ledger["event_id"] = (
        ledger["event_id"]
        .astype(str)
        .str.strip()
    )

    open_bets = ledger[
        ledger["status"]
        .astype(str)
        .str.upper()
        .eq("OPEN")
        &
        ledger["commence_time"].notna()
    ].copy()

    now = utc_now()

    open_bets["minutes_to_kickoff"] = (
        (
            open_bets["commence_time"]
            -
            now
        )
        .dt.total_seconds()
        /
        60.0
    )

    upcoming = open_bets[
        open_bets["minutes_to_kickoff"] > 0
    ].copy()

    upcoming = upcoming.sort_values(
        "minutes_to_kickoff"
    )

    print(f"UTC now: {now}")
    print(f"Open model bets: {len(open_bets)}")
    print()

    if upcoming.empty:

        print("No future OPEN bets.")
        return

    print("=" * 100)
    print("NEXT OPEN BETS")
    print("=" * 100)
    print()

    display = upcoming.head(10).copy()

    display["T_MINUS"] = (
        display["minutes_to_kickoff"]
        .map(
            lambda x:
            f"{x / 60:.2f}h"
        )
    )

    print(
        display[
            [
                "league",
                "home_team",
                "away_team",
                "bet_side",
                "commence_time",
                "T_MINUS",
            ]
        ].to_string(
            index=False
        )
    )

    history = load_history()

    due_rows = []

    # ========================================================
    # DETERMINE WHICH CHECKPOINTS ARE DUE
    # ========================================================

    for idx, bet in upcoming.iterrows():

        minutes = float(
            bet["minutes_to_kickoff"]
        )

        kickoff = bet[
            "commence_time"
        ]

        event_id = bet[
            "event_id"
        ]

        for checkpoint in CHECKPOINTS:

            start = checkpoint["start"]
            end = checkpoint["end"]

            if not (
                minutes <= start
                and
                minutes > end
            ):
                continue

            captured = (
                checkpoint_already_captured(
                    history=history,
                    event_id=event_id,
                    kickoff=kickoff,
                    start_minutes=start,
                    end_minutes=end,
                )
            )

            if captured:
                continue

            row = bet.copy()

            row[
                "checkpoint"
            ] = checkpoint["name"]

            due_rows.append(row)

    print()
    print("=" * 100)
    print("CHECKPOINT STATUS")
    print("=" * 100)
    print()

    if not due_rows:

        next_minutes = float(
            upcoming.iloc[0][
                "minutes_to_kickoff"
            ]
        )

        print(
            "No uncaptured closing-line "
            "checkpoint is currently due."
        )

        print(
            f"Next kickoff in "
            f"{next_minutes / 60:.2f} hours."
        )

        print()
        print(
            "No odds request made. "
            "API credits preserved."
        )

        return

    due = pd.DataFrame(
        due_rows
    )

    print(
        due[
            [
                "checkpoint",
                "league",
                "home_team",
                "away_team",
                "bet_side",
                "minutes_to_kickoff",
            ]
        ]
        .round(
            {
                "minutes_to_kickoff": 1
            }
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # ONE FETCH COVERS ALL CURRENTLY DUE EVENTS
    # ========================================================

    print()
    print("=" * 100)
    print("FETCHING ODDS SNAPSHOT")
    print("=" * 100)
    print()

    result = subprocess.run(
        [
            sys.executable,
            str(ODDS_FETCHER),
        ],
        cwd=str(ROOT),
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Odds fetcher failed."
        )

    # ========================================================
    # REBUILD CLV
    # ========================================================

    print()
    print("=" * 100)
    print("REBUILDING CLV REPORT")
    print("=" * 100)
    print()

    result = subprocess.run(
        [
            sys.executable,
            str(CLV_REPORT),
        ],
        cwd=str(ROOT),
    )

    if result.returncode != 0:

        raise RuntimeError(
            "CLV report failed."
        )

    print()
    print("=" * 100)
    print("CLOSING-LINE CHECK COMPLETE")
    print("=" * 100)
    print()

    print(
        "Snapshot captured for the "
        "currently due checkpoint(s)."
    )

    print(
        "Running this script again in the same "
        "checkpoint window will not fetch again."
    )


if __name__ == "__main__":
    main()
